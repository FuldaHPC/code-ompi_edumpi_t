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
 * Coverage for ompi.coll_algorithm_selected.  Needs at least two processes
 * (skips with one).  Runs a broadcast, an allreduce and a barrier on a
 * duplicate of MPI_COMM_WORLD and checks that every operation reports one
 * event for this communicator with the right operation, communicator size
 * and message size.  If the environment variable EXPECT_BCAST_ALGORITHM is
 * set (for example together with --mca coll_tuned_use_dynamic_rules 1
 * --mca coll_tuned_bcast_algorithm 6) the id of the broadcast event must
 * equal it.  Prints every event for inspection.
 */

#include "opal_config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define OP_ALLREDUCE 2
#define OP_BARRIER 6
#define OP_BCAST 7
#define MAXEV 64

typedef struct {
    int64_t msg_size;
    uint64_t parent;
    int32_t operation, algorithm, component, comm_size, root, level;
} ev_t;

static ev_t evs[MAXEV];
static int nev = 0, failures = 0;

static void expect(const char *what, int cond)
{
    printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) {
        ++failures;
    }
}

static void cb(MPI_T_event_instance ev, MPI_T_event_registration h __opal_attribute_unused__,
               MPI_T_cb_safety s __opal_attribute_unused__, void *u __opal_attribute_unused__)
{
    ev_t *e;
    if (nev >= MAXEV) {
        return;
    }
    e = &evs[nev++];
    /* request, msg_size, parent_comm, operation, algorithm, component, comm_size, root, level */
    (void) MPI_T_event_read(ev, 1, &e->msg_size);
    (void) MPI_T_event_read(ev, 2, &e->parent);
    (void) MPI_T_event_read(ev, 3, &e->operation);
    (void) MPI_T_event_read(ev, 4, &e->algorithm);
    (void) MPI_T_event_read(ev, 5, &e->component);
    (void) MPI_T_event_read(ev, 6, &e->comm_size);
    (void) MPI_T_event_read(ev, 7, &e->root);
    (void) MPI_T_event_read(ev, 8, &e->level);
}

/* msg < 0: any message size (internal collectives of communicator creation have other sizes) */
static const ev_t *find(int op, int64_t msg)
{
    for (int i = 0; i < nev; ++i) {
        if (op == evs[i].operation && 0 == evs[i].level && (msg < 0 || msg == evs[i].msg_size)) {
            return &evs[i];
        }
    }
    return NULL;
}

int main(int argc, char **argv)
{
    int provided, idx, size, rank, n = 1024;
    MPI_Comm comm;
    MPI_T_event_registration reg;
    double *buf, *out;
    const ev_t *e;
    const char *expect_alg = getenv("EXPECT_BCAST_ALGORITHM");

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2) {
        printf("SKIP: needs at least 2 processes\n");
        MPI_Finalize();
        return 77;
    }
    if (MPI_SUCCESS != MPI_T_init_thread(MPI_THREAD_SINGLE, &provided)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.coll_algorithm_selected", &idx)) {
        printf("SKIP: ompi.coll_algorithm_selected not available\n");
        MPI_Finalize();
        return 77;
    }

    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    expect("alloc handle", MPI_SUCCESS == MPI_T_event_handle_alloc(idx, &comm, MPI_INFO_NULL, &reg));
    (void) MPI_T_event_register_callback(reg, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, NULL, cb);

    buf = calloc(n, sizeof(double));
    out = calloc(n, sizeof(double));
    MPI_Bcast(buf, n, MPI_DOUBLE, 0, comm);
    MPI_Allreduce(buf, out, n, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Barrier(comm);
    {
        MPI_Request r;
        MPI_Ibcast(buf, n, MPI_DOUBLE, 0, comm, &r);
        MPI_Wait(&r, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < nev; ++i) {
        printf("EVENT rank %d op %d alg %d component %d size %d root %d level %d parent %llu msg %lld\n",
               rank, evs[i].operation, evs[i].algorithm, evs[i].component, evs[i].comm_size,
               evs[i].root, evs[i].level, (unsigned long long) evs[i].parent,
               (long long) evs[i].msg_size);
    }
    e = find(OP_BCAST, (int64_t) n * 8);
    expect("bcast event", NULL != e);
    if (e) {
        expect("bcast: comm size, root 0, message size", e->comm_size == size && 0 == e->root
                                                           && (int64_t) n * 8 == e->msg_size);
        if (expect_alg) {
            expect("bcast: algorithm id as forced", e->algorithm == atoi(expect_alg));
        }
    }
    e = find(OP_ALLREDUCE, (int64_t) n * 8);
    expect("allreduce event", NULL != e);
    if (e) {
        expect("allreduce: comm size, no root, message size",
               e->comm_size == size && -1 == e->root && (int64_t) n * 8 == e->msg_size);
    }
    expect("barrier event", NULL != find(OP_BARRIER, 0));

    (void) MPI_T_event_handle_free(reg, NULL, NULL);
    MPI_Comm_free(&comm);
    free(buf);
    free(out);
    MPI_T_finalize();
    MPI_Finalize();
    if (0 == rank) {
        printf("RESULT: %s (%d failures)\n", 0 == failures ? "PASS" : "FAIL", failures);
    }
    return 0 == failures ? 0 : 1;
}
