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
 * PML-independent MPI_T events for point-to-point operations.
 *
 * Four event types describe a message from the user's point of view,
 * whatever the PML (ob1, ucx, cm, ...) does underneath:
 *
 *   ompi.pml_send_start      a send was issued   (peer, tag, length, mode)
 *   ompi.pml_recv_start      a receive was posted (peer or MPI_ANY_SOURCE, tag, capacity)
 *   ompi.pml_send_complete   a send finished     (peer, tag, bytes)
 *   ompi.pml_recv_complete   a receive finished  (actual source, tag, bytes)
 *
 * All of them are bound to MPI_T_BIND_MPI_COMM.  The request element is an
 * opaque correlator (the PML's request address, never to be dereferenced);
 * it is 0 for an operation that has no request (blocking fast path).  The
 * start and complete event of one request carry the same token.
 *
 * A negative tag marks a message that Open MPI sends internally on behalf of
 * a collective operation.
 *
 * The ob1 specific events (pml_ob1_*) remain available for tools that want
 * the matching-engine detail; these four are what a tool needs to account
 * for messages and their envelopes independent of the PML in use.
 *
 * Raising costs one load and a predicted branch when nobody listens.
 */

#ifndef MCA_PML_BASE_EVENTS_H
#define MCA_PML_BASE_EVENTS_H

#include "ompi_config.h"

#include <assert.h>

#include "opal/mca/base/mca_base_event.h"
#include "ompi/communicator/communicator.h"
#include "ompi/proc/proc.h"

BEGIN_C_DECLS

enum {
    MCA_PML_BASE_EVENT_SEND_START,
    MCA_PML_BASE_EVENT_RECV_START,
    MCA_PML_BASE_EVENT_SEND_COMPLETE,
    MCA_PML_BASE_EVENT_RECV_COMPLETE,
    MCA_PML_BASE_EVENT_MAX,
};

/* flags of the start events */
enum {
    MCA_PML_BASE_EVENT_FLAG_SYNC       = 0x01, /* synchronous mode send */
    MCA_PML_BASE_EVENT_FLAG_READY      = 0x02, /* ready mode send */
    MCA_PML_BASE_EVENT_FLAG_BUFFERED   = 0x04, /* buffered mode send */
    MCA_PML_BASE_EVENT_FLAG_BLOCKING   = 0x08, /* blocking call (MPI_Send/MPI_Recv) */
    MCA_PML_BASE_EVENT_FLAG_PERSISTENT = 0x10, /* started with MPI_Start */
};

/* flags of the complete events */
enum {
    MCA_PML_BASE_EVENT_COMPLETE_CANCELLED = 0x01,
    MCA_PML_BASE_EVENT_COMPLETE_ERROR     = 0x02,
    MCA_PML_BASE_EVENT_COMPLETE_TRUNCATED = 0x04,
};

/* send_start, recv_start */
struct mca_pml_base_start_event_t {
    uint64_t request;   /* opaque; 0: the operation has no request */
    uint64_t parent;    /* enclosing collective request; 0: none */
    int64_t length;     /* bytes to send, or capacity of the receive buffer */
    int32_t peer;       /* rank in the bound communicator (MPI_ANY_SOURCE possible) */
    int32_t peer_world; /* rank in the job's MPI_COMM_WORLD, -1: unknown */
    int32_t tag;        /* negative: internal message of a collective */
    int32_t flags;      /* MCA_PML_BASE_EVENT_FLAG_* */
};
typedef struct mca_pml_base_start_event_t mca_pml_base_start_event_t;

/* send_complete, recv_complete */
struct mca_pml_base_complete_event_t {
    uint64_t request;
    int64_t length;     /* bytes transferred */
    int32_t peer;       /* receive: actual source */
    int32_t peer_world;
    int32_t tag;        /* receive: actual tag */
    int32_t flags;      /* MCA_PML_BASE_EVENT_COMPLETE_* */
};
typedef struct mca_pml_base_complete_event_t mca_pml_base_complete_event_t;

OMPI_DECLSPEC extern mca_base_event_t *mca_pml_base_event[MCA_PML_BASE_EVENT_MAX];

/* Token of the collective request on whose behalf the calling thread is posting point-to-point messages (a
   non-blocking collective schedule), 0 if none. Set by the collective component around its posting loop; copied
   into the "parent" element of the start events so that a tool can attribute the messages to the request. */
OMPI_DECLSPEC extern _Thread_local uint64_t mca_pml_base_event_parent;

/* Called from the PML framework's register function. */
void mca_pml_base_events_register(void);

static inline bool mca_pml_base_event_wanted(int which)
{
    assert(which >= 0 && which < MCA_PML_BASE_EVENT_MAX);
    const mca_base_event_t *event = mca_pml_base_event[which];

    return NULL != event && mca_base_event_active(event);
}

/* rank in MPI_COMM_WORLD of a peer, -1 for MPI_ANY_SOURCE or on failure */
static inline int32_t mca_pml_base_event_peer_world(ompi_communicator_t *comm, int peer)
{
    ompi_proc_t *proc;

    if (peer < 0 || NULL == comm) {
        return -1;
    }
    proc = ompi_comm_peer_lookup(comm, peer);
    return NULL == proc ? -1 : (int32_t) proc->super.proc_name.vpid;
}

/* which: MCA_PML_BASE_EVENT_SEND_START or MCA_PML_BASE_EVENT_RECV_START */
static inline void mca_pml_base_event_raise_start(int which, ompi_communicator_t *comm,
                                                  const void *request, int peer, int tag,
                                                  size_t length, int flags)
{
    if (OPAL_LIKELY(!mca_pml_base_event_wanted(which))) {
        return;
    }

    mca_pml_base_start_event_t payload = {.request = (uint64_t) (uintptr_t) request,
                                          .parent = mca_pml_base_event_parent,
                                          .length = (int64_t) length,
                                          .peer = peer,
                                          .peer_world = mca_pml_base_event_peer_world(comm, peer),
                                          .tag = tag,
                                          .flags = flags};
    mca_base_event_raise_bound(mca_pml_base_event[which], NULL, comm, &payload);
}

/* which: MCA_PML_BASE_EVENT_SEND_COMPLETE or MCA_PML_BASE_EVENT_RECV_COMPLETE */
static inline void mca_pml_base_event_raise_complete(int which, ompi_communicator_t *comm,
                                                     const void *request, int peer, int tag,
                                                     size_t length, int flags)
{
    if (OPAL_LIKELY(!mca_pml_base_event_wanted(which))) {
        return;
    }

    mca_pml_base_complete_event_t payload = {.request = (uint64_t) (uintptr_t) request,
                                             .length = (int64_t) length,
                                             .peer = peer,
                                             .peer_world = mca_pml_base_event_peer_world(comm, peer),
                                             .tag = tag,
                                             .flags = flags};
    mca_base_event_raise_bound(mca_pml_base_event[which], NULL, comm, &payload);
}

END_C_DECLS

#endif /* MCA_PML_BASE_EVENTS_H */
