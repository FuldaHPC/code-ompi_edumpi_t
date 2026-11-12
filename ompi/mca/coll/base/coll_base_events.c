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

#include "ompi_config.h"

#include <stddef.h>

#include "ompi/mca/coll/base/coll_base_events.h"
#include "opal/mca/base/mca_base_pvar.h"
#include "opal/mca/base/mca_base_var.h"
#include "opal/class/opal_hash_table.h"
#include "opal/mca/threads/mutex.h"

mca_base_event_t *mca_coll_base_event_algorithm = NULL;

/* helper communicators of hierarchical components: comm pointer -> {parent, level} */
typedef struct {
    uint64_t parent;
    int32_t level;
} helper_info_t;
static opal_hash_table_t helper_table;
static opal_mutex_t helper_lock = OPAL_MUTEX_STATIC_INIT;
static bool helper_table_ready = false;

void mca_coll_base_event_helper_add(ompi_communicator_t *helper, ompi_communicator_t *parent,
                                    int level)
{
    helper_info_t *info;

    if (NULL == helper) {
        return;
    }
    OPAL_THREAD_LOCK(&helper_lock);
    if (!helper_table_ready) {
        OBJ_CONSTRUCT(&helper_table, opal_hash_table_t);
        opal_hash_table_init(&helper_table, 64);
        helper_table_ready = true;
    }
    info = malloc(sizeof(*info));
    if (NULL != info) {
        void *old = NULL;
        info->parent = (uint64_t) (uintptr_t) parent;
        info->level = level;
        if (OPAL_SUCCESS == opal_hash_table_get_value_uint64(&helper_table, (uint64_t) (uintptr_t) helper, &old)) {
            free(old);
        }
        opal_hash_table_set_value_uint64(&helper_table, (uint64_t) (uintptr_t) helper, info);
    }
    OPAL_THREAD_UNLOCK(&helper_lock);
}

void mca_coll_base_event_helper_remove(ompi_communicator_t *helper)
{
    void *old = NULL;

    OPAL_THREAD_LOCK(&helper_lock);
    if (helper_table_ready
        && OPAL_SUCCESS == opal_hash_table_get_value_uint64(&helper_table, (uint64_t) (uintptr_t) helper, &old)) {
        opal_hash_table_remove_value_uint64(&helper_table, (uint64_t) (uintptr_t) helper);
        free(old);
    }
    OPAL_THREAD_UNLOCK(&helper_lock);
}

bool mca_coll_base_event_helper_lookup(const ompi_communicator_t *comm, uint64_t *parent_comm,
                                       int32_t *level)
{
    void *found = NULL;
    bool ok = false;

    OPAL_THREAD_LOCK(&helper_lock);
    if (helper_table_ready
        && OPAL_SUCCESS == opal_hash_table_get_value_uint64(&helper_table, (uint64_t) (uintptr_t) comm, &found)
        && NULL != found) {
        *parent_comm = ((helper_info_t *) found)->parent;
        *level = ((helper_info_t *) found)->level;
        ok = true;
    }
    OPAL_THREAD_UNLOCK(&helper_lock);
    return ok;
}

#define COLL_BASE_TICKS_PER_SECOND ((opal_count_t) 1000000000)

static const mca_base_var_type_t algorithm_types[] = {
    MCA_BASE_VAR_TYPE_UINT64_T, MCA_BASE_VAR_TYPE_INT64_T, MCA_BASE_VAR_TYPE_UINT64_T,
    MCA_BASE_VAR_TYPE_INT32_T,  MCA_BASE_VAR_TYPE_INT32_T, MCA_BASE_VAR_TYPE_INT32_T,
    MCA_BASE_VAR_TYPE_INT32_T,  MCA_BASE_VAR_TYPE_INT32_T, MCA_BASE_VAR_TYPE_INT32_T};
static const ptrdiff_t algorithm_offsets[] = {
    offsetof(mca_coll_base_algorithm_event_t, request),
    offsetof(mca_coll_base_algorithm_event_t, msg_size),
    offsetof(mca_coll_base_algorithm_event_t, parent_comm),
    offsetof(mca_coll_base_algorithm_event_t, operation),
    offsetof(mca_coll_base_algorithm_event_t, algorithm),
    offsetof(mca_coll_base_algorithm_event_t, component),
    offsetof(mca_coll_base_algorithm_event_t, comm_size),
    offsetof(mca_coll_base_algorithm_event_t, root),
    offsetof(mca_coll_base_algorithm_event_t, level)};

void mca_coll_base_events_register(void)
{
    const int *register_producers = NULL;
    mca_base_event_source_t *source = NULL;
    int source_index, var_index, event_index;

    mca_coll_base_event_algorithm = NULL;

    var_index = mca_base_var_find("opal", "mca", "base", "event_register_producers");
    if (0 <= var_index) {
        (void) mca_base_var_get_value(var_index, &register_producers, NULL, NULL);
    }
    if (NULL != register_producers && 0 == register_producers[0]) {
        return;
    }

    source_index = mca_base_event_source_register("ompi", "Open MPI runtime events (ordered)",
                                                  MCA_BASE_EVENT_SOURCE_ORDERED,
                                                  COLL_BASE_TICKS_PER_SECOND, OPAL_COUNT_MAX, NULL,
                                                  true, NULL);
    if (0 > source_index
        || OPAL_SUCCESS != mca_base_event_source_get_by_index(source_index, &source)) {
        return;
    }

    /* full name: ompi.coll_algorithm_selected */
    event_index = mca_base_event_register(
        "ompi", "coll", NULL, "algorithm_selected",
        "A collective component selected an algorithm for an operation on this communicator. "
        "Elements: request, msg_size (bytes per process), parent_comm, operation (COLLTYPE), "
        "algorithm (component specific id; for tuned and libnbc see the enumeration of the "
        "control variable <component>_<operation>_algorithm), component, comm_size, root "
        "(-1: none), level (0 user communicator, 1 inter-node, 2 intra-node helper communicator "
        "of a hierarchical component).",
        OPAL_INFO_LVL_4, 9, algorithm_types, algorithm_offsets, NULL, MCA_BASE_VAR_BIND_MPI_COMM,
        MCA_BASE_EVENT_FLAG_IWG, source, NULL);
    if (0 > event_index) {
        return;
    }
    (void) mca_base_event_get_by_index(event_index, &mca_coll_base_event_algorithm);
}
