/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      Jeffrey M. Squyres.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 * SPDX-License-Identifier: BSD-3-Clause-Open-MPI
 *
 * Registration of Open MPI's in-tree MPI_T event producers, and the event
 * type handles the producer raise sites use.  See specs/mpi-t-events/spec.md
 * section 7.
 */

#ifndef OMPI_MPIT_EVENTS_H
#define OMPI_MPIT_EVENTS_H

#include "ompi_config.h"

#include "opal/mca/base/mca_base_event.h"

struct ompi_communicator_t;

BEGIN_C_DECLS

/* The "model" element of the ompi.mpi.initialization / ompi.mpi.finalization
   event payloads uses the public OMPI_T_MODEL_* values from <mpi.h>
   (OMPI_T_MODEL_WORLD / OMPI_T_MODEL_SESSION).  The world_rank / world_size
   elements are meaningful only for the world model (a process has no rank in a
   session) and are -1 otherwise. */

/* Which MPI ABI a registering MPI_T tool is using.  This governs the
   representation of the MPI object handles carried in event payloads
   (communicator, window, error-handler, session): the Open MPI ABI uses the
   internal object pointer; the MPI Standard ABI uses an integer handle.  A
   process links exactly one ABI, so this is process-global; it is set by
   MPI_T_event_register_callback() and read by the producer raise sites. */
typedef enum {
    OMPI_MPIT_ABI_OMPI = 0,     /* Open MPI ABI: handle == internal object pointer */
    OMPI_MPIT_ABI_STANDARD = 1  /* MPI Standard ABI: handle == integer handle */
} ompi_mpit_abi_t;

/* Hard-coded to the Open MPI ABI for now.  When the MPI Standard ABI lands
   (open-mpi/ompi#13280), its MPI_T_event_register_callback entry point will set
   this to OMPI_MPIT_ABI_STANDARD, and the producers' "else" branches (marked
   "TODO ABI") will fill in the Standard-ABI handle values. */
OMPI_DECLSPEC extern ompi_mpit_abi_t ompi_mpit_callback_abi;

/* Event type handles for the in-tree producers.  NULL until (and unless) the
   producers are registered, so a raise site must NULL-check before raising. */
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_comm_created;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_comm_freed;

/* Set REQUEST (generic request states, bound to a communicator): ompi.mpi.request_activated and
   ompi.mpi.request_completed, semantic names REQUEST_ACTIVATED / REQUEST_COMPLETED of the MPI Forum Tools WG
   "entities and sets" side document. Payload: request (u64 token = the MPI_Request handle), comm (u64), kind (i32,
   OMPI_REQUEST_KIND_*), flags (i32, OMPI_REQUEST_FLAG_*). Raised for non-blocking collective requests (libnbc);
   point-to-point requests additionally report their envelope with the pml_* events. A token is valid from
   REQUEST_ACTIVATED to REQUEST_COMPLETED of one activation. */
/* Set HANDLE for requests (not bound to an object): ompi.mpi.request_created / ompi.mpi.request_freeing, semantic names
   REQUEST_CREATED / REQUEST_FREEING. Payload: request (u64, the MPI_Request handle), comm (u64), kind (i32,
   OMPI_REQUEST_KIND_*), flags (i32, OMPI_REQUEST_FLAG_*). Raised for requests that are visible to the user (non-blocking
   calls and persistent requests), never for the internal requests of blocking calls. A request that completes
   inline and is represented by a shared pre-completed request object (the "dummy request") has no created and no
   freeing event. REQUEST_FREEING is raised before the request is released, so that a request-bound PVAR handle can
   still be read; the token is invalid afterwards. */
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_request_created;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_request_freeing;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_request_activated;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_request_completed;

enum {
    OMPI_REQUEST_KIND_UNKNOWN = -1,   /* only in REQUEST_FREEING, where the owner does not keep the kind */
    OMPI_REQUEST_KIND_PML_SEND = 0,
    OMPI_REQUEST_KIND_PML_RECV = 1,
    OMPI_REQUEST_KIND_COLL = 2,
    OMPI_REQUEST_KIND_OSC = 3,
    OMPI_REQUEST_KIND_IO = 4,
    OMPI_REQUEST_KIND_GENERALIZED = 5,
    OMPI_REQUEST_KIND_PART = 6,
};
enum {
    OMPI_REQUEST_FLAG_PERSISTENT = 0x1,
};

struct ompi_mpit_request_event_t {
    uint64_t request;
    uint64_t comm;
    int32_t kind;
    int32_t flags;
};
typedef struct ompi_mpit_request_event_t ompi_mpit_request_event_t;
/* Object-bound (MPI_T_BIND_MPI_COMM): raised for the specific communicator being
   named, so only registrations bound to that communicator are notified. */
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_comm_name_set;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_initialization;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_finalization;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_errhandler_invoked;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_win_created;
OMPI_DECLSPEC extern mca_base_event_t *ompi_event_win_freed;

/* Register all in-tree MPI_T event producers (sources + event types).  This is
   the single core-producer entry point; it is idempotent and one-shot, so it
   may be called from every path that registers MCA parameters (ompi_info,
   instance init, MPI_T_init_thread).  Gated by the master MCA parameter
   mca_base_event_register_producers (default on). */
OMPI_DECLSPEC void ompi_mpit_register_events(void);

static inline void ompi_mpit_raise_request_handle(int created, struct ompi_communicator_t *comm, const void *request,
                                                  int kind, int flags)
{
    mca_base_event_t *event = created ? ompi_event_request_created : ompi_event_request_freeing;

    if (OPAL_LIKELY(NULL == event || !mca_base_event_active(event))) {
        return;
    }
    ompi_mpit_request_event_t payload = {.request = (uint64_t) (uintptr_t) request,
                                         .comm = (uint64_t) (uintptr_t) comm,
                                         .kind = kind,
                                         .flags = flags};
    mca_base_event_raise(event, NULL, &payload);
}

/* raise REQUEST_ACTIVATED (activated != 0) or REQUEST_COMPLETED for a request; free when nobody listens */
static inline void ompi_mpit_raise_request_state(int activated, struct ompi_communicator_t *comm, const void *request,
                                                 int kind, int flags)
{
    mca_base_event_t *event = activated ? ompi_event_request_activated : ompi_event_request_completed;

    if (OPAL_LIKELY(NULL == event || !mca_base_event_active(event))) {
        return;
    }
    ompi_mpit_request_event_t payload = {.request = (uint64_t) (uintptr_t) request,
                                         .comm = (uint64_t) (uintptr_t) comm,
                                         .kind = kind,
                                         .flags = flags};
    mca_base_event_raise_bound(event, NULL, comm, &payload);
}

END_C_DECLS

#endif /* OMPI_MPIT_EVENTS_H */
