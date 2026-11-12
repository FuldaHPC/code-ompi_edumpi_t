/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026      Hochschule Fulda.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 * SPDX-License-Identifier: BSD-3-Clause-Open-MPI
 *
 * Coverage for the PML independent point-to-point events
 * ompi.pml_send_start, ompi.pml_recv_start, ompi.pml_send_complete and
 * ompi.pml_recv_complete.  A rank sends to itself over MPI_COMM_SELF
 * (non-blocking, then blocking) and the test checks that the envelope the
 * events report -- peer, tag, byte count -- is what the program used.  The
 * checks hold for every PML; send_complete of a non-blocking send is only
 * reported, because not every PML can raise it (see pml_base_events.h).
 */

#include "opal_config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NEV 4
static const char *names[NEV] = {"ompi.pml_send_start", "ompi.pml_recv_start",
                                 "ompi.pml_send_complete", "ompi.pml_recv_complete"};
enum { SEND_START, RECV_START, SEND_COMPLETE, RECV_COMPLETE };

typedef struct {
    int fired;
    int64_t length;
    int32_t peer, tag, flags;
} seen_t;

static seen_t seen[NEV];
static int failures = 0;

static void expect(const char *what, int cond)
{
    printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) {
        ++failures;
    }
}

static void cb(MPI_T_event_instance ev, MPI_T_event_registration h __opal_attribute_unused__,
               MPI_T_cb_safety s __opal_attribute_unused__, void *user)
{
    int which = (int) (intptr_t) user;
    seen_t *e = &seen[which];
    /* start: request, parent, length, peer, peer_world, tag, flags
       complete: request, length, peer, peer_world, tag, flags */
    int il, ip, it, ifl;
    if (SEND_START == which || RECV_START == which) {
        il = 2; ip = 3; it = 5; ifl = 6;
    } else {
        il = 1; ip = 2; it = 4; ifl = 5;
    }
    ++e->fired;
    (void) MPI_T_event_read(ev, il, &e->length);
    (void) MPI_T_event_read(ev, ip, &e->peer);
    (void) MPI_T_event_read(ev, it, &e->tag);
    (void) MPI_T_event_read(ev, ifl, &e->flags);
}

static void reset(void)
{
    memset(seen, 0, sizeof(seen));
}

int main(int argc, char **argv)
{
    int provided, idx[NEV], i, in[4] = {1, 2, 3, 4}, out[4] = {0, 0, 0, 0};
    MPI_Comm self = MPI_COMM_SELF;
    MPI_T_event_registration reg[NEV];
    MPI_Request sreq, rreq;

    if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
        printf("SKIP: MPI_Init failed (no singleton support here)\n");
        return 77;
    }
    if (MPI_SUCCESS != MPI_T_init_thread(MPI_THREAD_SINGLE, &provided)) {
        printf("SKIP: MPI_T_init_thread failed\n");
        MPI_Finalize();
        return 77;
    }
    for (i = 0; i < NEV; ++i) {
        if (MPI_SUCCESS != MPI_T_event_get_index(names[i], &idx[i])) {
            printf("SKIP: %s not registered\n", names[i]);
            MPI_T_finalize();
            MPI_Finalize();
            return 77;
        }
        expect("register handle", MPI_SUCCESS == MPI_T_event_handle_alloc(idx[i], &self, MPI_INFO_NULL, &reg[i]));
        (void) MPI_T_event_register_callback(reg[i], MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL,
                                             (void *) (intptr_t) i, cb);
    }

    /* non-blocking pair */
    reset();
    MPI_Irecv(out, 4, MPI_INT, 0, 7, self, &rreq);
    MPI_Isend(in, 4, MPI_INT, 0, 7, self, &sreq);
    MPI_Wait(&sreq, MPI_STATUS_IGNORE);
    MPI_Wait(&rreq, MPI_STATUS_IGNORE);
    expect("data arrived", 0 == memcmp(in, out, sizeof(in)));
    expect("send_start once", 1 == seen[SEND_START].fired);
    expect("send_start envelope (peer 0, tag 7, 16 bytes)",
           0 == seen[SEND_START].peer && 7 == seen[SEND_START].tag && 16 == seen[SEND_START].length);
    expect("recv_start once", 1 == seen[RECV_START].fired);
    expect("recv_start envelope (peer 0, tag 7, capacity 16)",
           0 == seen[RECV_START].peer && 7 == seen[RECV_START].tag && 16 == seen[RECV_START].length);
    expect("recv_complete once", 1 == seen[RECV_COMPLETE].fired);
    expect("recv_complete envelope (source 0, tag 7, 16 bytes)",
           0 == seen[RECV_COMPLETE].peer && 7 == seen[RECV_COMPLETE].tag
               && 16 == seen[RECV_COMPLETE].length);
    printf("INFO: send_complete fired %d times for the non-blocking send\n", seen[SEND_COMPLETE].fired);

    /* blocking pair, message smaller than the buffer, synchronous send flag */
    reset();
    MPI_Irecv(out, 4, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, self, &rreq);
    MPI_Ssend(in, 2, MPI_INT, 0, 9, self);
    MPI_Wait(&rreq, MPI_STATUS_IGNORE);
    expect("any-source recv_start reports MPI_ANY_SOURCE", MPI_ANY_SOURCE == seen[RECV_START].peer);
    expect("blocking send_start flags: blocking + synchronous",
           (seen[SEND_START].flags & 0x08) && (seen[SEND_START].flags & 0x01));
    expect("recv_complete reports the real source and tag and 8 bytes",
           0 == seen[RECV_COMPLETE].peer && 9 == seen[RECV_COMPLETE].tag
               && 8 == seen[RECV_COMPLETE].length);
    expect("blocking send_complete fired", 1 == seen[SEND_COMPLETE].fired);

    for (i = 0; i < NEV; ++i) {
        (void) MPI_T_event_handle_free(reg[i], NULL, NULL);
    }
    MPI_T_finalize();
    MPI_Finalize();

    printf("RESULT: %s (%d failures)\n", 0 == failures ? "PASS" : "FAIL", failures);
    return 0 == failures ? 0 : 1;
}
