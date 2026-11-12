/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      Hochschule Fulda.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 * SPDX-License-Identifier: BSD-3-Clause-Open-MPI
 */

/*
 * Request-bound MPI_T performance variables of the libnbc component (set REQUEST of the MPI Forum Tools WG
 * "entities and sets" side document):
 *
 *   coll_libnbc_request_messages   messages the request's schedule sent, one counter per peer of the communicator
 *   coll_libnbc_request_bytes      the bytes of those messages, one counter per peer
 *
 * Both are bound to an MPI_Request.  A tool allocates the handles in the handler of ompi.mpi.request_created and reads
 * them in the handler of ompi.mpi.request_freeing; this is how a tool follows the internal messages of one
 * non-blocking collective although other operations run on the same communicator meanwhile.  The storage is a side
 * table, created when a handle is bound and removed at the last unbind, so a request nobody binds to costs one
 * load and a predicted branch per message.
 */

#include "ompi_config.h"

#include "coll_libnbc.h"
#include "nbc_internal.h"

#include "opal/mca/base/mca_base_pvar.h"
#include "opal/mca/threads/mutex.h"
#include "ompi/communicator/communicator.h"
#include "ompi/datatype/ompi_datatype.h"

volatile int ompi_coll_libnbc_request_pvars_bound = 0;

typedef struct {
    const void *request;
    int size;
    int refs;
    unsigned long long *messages;
    unsigned long long *bytes;
} request_entry_t;

static request_entry_t *entries = NULL;
static int n_entries = 0, cap_entries = 0;
static opal_mutex_t entries_lock = OPAL_MUTEX_STATIC_INIT;

static request_entry_t *find_entry(const void *request)
{
    for (int i = 0; i < n_entries; ++i) {
        if (entries[i].request == request) {
            return &entries[i];
        }
    }
    return NULL;
}

static int peers_of(ompi_communicator_t *comm)
{
    return OMPI_COMM_IS_INTER(comm) ? ompi_comm_remote_size(comm) : ompi_comm_size(comm);
}

static int request_notify(mca_base_pvar_t *pvar, mca_base_pvar_event_t event, void *obj_handle, int *count)
{
    ompi_coll_libnbc_request_t *req = (ompi_coll_libnbc_request_t *) obj_handle;
    request_entry_t *e;
    int rc = OMPI_SUCCESS;

    (void) pvar;
    if (MCA_BASE_PVAR_HANDLE_BIND == event) {
        if (NULL == req || !ompi_coll_libnbc_is_request(&req->super.super)) {
            return OPAL_ERR_BAD_PARAM;                 /* not a libnbc request: other layouts, refuse */
        }
        OPAL_THREAD_LOCK(&entries_lock);
        e = find_entry(req);
        if (NULL == e) {
            if (n_entries == cap_entries) {
                int ncap = cap_entries ? 2 * cap_entries : 16;
                request_entry_t *ne = realloc(entries, ncap * sizeof(*entries));
                if (NULL == ne) {
                    OPAL_THREAD_UNLOCK(&entries_lock);
                    return OPAL_ERR_OUT_OF_RESOURCE;
                }
                entries = ne;
                cap_entries = ncap;
            }
            e = &entries[n_entries];
            e->request = req;
            e->size = peers_of(req->comm);
            e->refs = 0;
            e->messages = calloc(e->size ? e->size : 1, sizeof(unsigned long long));
            e->bytes = calloc(e->size ? e->size : 1, sizeof(unsigned long long));
            if (NULL == e->messages || NULL == e->bytes) {
                free(e->messages);
                free(e->bytes);
                OPAL_THREAD_UNLOCK(&entries_lock);
                return OPAL_ERR_OUT_OF_RESOURCE;
            }
            ++n_entries;
            ompi_coll_libnbc_request_pvars_bound = n_entries;
        }
        ++e->refs;
        *count = e->size;
        OPAL_THREAD_UNLOCK(&entries_lock);
    } else if (MCA_BASE_PVAR_HANDLE_UNBIND == event) {
        OPAL_THREAD_LOCK(&entries_lock);
        e = find_entry(obj_handle);
        if (NULL != e && 0 == --e->refs) {
            free(e->messages);
            free(e->bytes);
            *e = entries[--n_entries];
            ompi_coll_libnbc_request_pvars_bound = n_entries;
        }
        OPAL_THREAD_UNLOCK(&entries_lock);
    }
    return rc;
}

static int get_request_array(void *value, void *obj_handle, int bytes)
{
    request_entry_t *e;
    int rc = OPAL_ERR_BAD_PARAM;

    OPAL_THREAD_LOCK(&entries_lock);
    e = find_entry(obj_handle);
    if (NULL != e) {
        memcpy(value, bytes ? e->bytes : e->messages, e->size * sizeof(unsigned long long));
        rc = OMPI_SUCCESS;
    }
    OPAL_THREAD_UNLOCK(&entries_lock);
    return rc;
}

static int get_request_messages(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle)
{
    (void) pvar;
    return get_request_array(value, obj_handle, 0);
}

static int get_request_bytes(const struct mca_base_pvar_t *pvar, void *value, void *obj_handle)
{
    (void) pvar;
    return get_request_array(value, obj_handle, 1);
}

/* called from NBC_Start_round for every send of a schedule, only while a handle is bound to some request */
void ompi_coll_libnbc_request_account_send(NBC_Handle *handle, int dest, size_t bytes)
{
    request_entry_t *e;

    OPAL_THREAD_LOCK(&entries_lock);
    e = find_entry(handle);
    if (NULL != e && dest >= 0 && dest < e->size) {
        e->messages[dest] += 1;
        e->bytes[dest] += bytes;
    }
    OPAL_THREAD_UNLOCK(&entries_lock);
}

void ompi_coll_libnbc_request_pvars_register(void)
{
    (void) mca_base_component_pvar_register(&mca_coll_libnbc_component.super.collm_version, "request_messages",
                                            "Messages the schedule of a non-blocking collective request sent, per "
                                            "peer of the communicator (bound to an MPI_Request; allocate the handle in "
                                            "the ompi.mpi.request_created handler, read it in ompi.mpi.request_freeing)",
                                            OPAL_INFO_LVL_4, MPI_T_PVAR_CLASS_COUNTER,
                                            MCA_BASE_VAR_TYPE_UNSIGNED_LONG_LONG, NULL, MPI_T_BIND_MPI_REQUEST,
                                            MCA_BASE_PVAR_FLAG_READONLY | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                            get_request_messages, NULL, request_notify, NULL);
    (void) mca_base_component_pvar_register(&mca_coll_libnbc_component.super.collm_version, "request_bytes",
                                            "Bytes of the messages the schedule of a non-blocking collective request "
                                            "sent, per peer of the communicator (bound to an MPI_Request)",
                                            OPAL_INFO_LVL_4, MPI_T_PVAR_CLASS_COUNTER,
                                            MCA_BASE_VAR_TYPE_UNSIGNED_LONG_LONG, NULL, MPI_T_BIND_MPI_REQUEST,
                                            MCA_BASE_PVAR_FLAG_READONLY | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                            get_request_bytes, NULL, request_notify, NULL);
}
