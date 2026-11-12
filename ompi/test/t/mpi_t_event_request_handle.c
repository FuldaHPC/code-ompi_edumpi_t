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
 * Request lifetime test (the problem of the MPI Forum Tools WG "entities and sets" side document): a tool cannot bind
 * to a short-lived request after the call returned, so it learns of every request from ompi.mpi.request_created and
 * ompi.mpi.request_freeing.  A loop of non-blocking one-byte operations (MPI_Isend/MPI_Irecv pairs and, with at
 * least two processes, MPI_Ibcast) must produce exactly one created and one freeing event for each request, with the
 * same token, and the freeing event must come while the request still exists.  Persistent requests are covered by
 * a second loop.  Arguments: [iterations] (default 20000).
 */

#include "opal_config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long n_created = 0, n_freeing = 0, n_unmatched = 0, n_bad_kind = 0;
#define TABLE 65536
static uint64_t live[TABLE];       /* tokens that were created and not yet freed */
static int failures = 0;

static void expect(const char *what, int cond)
{
    printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) {
        ++failures;
    }
}

static size_t slot(uint64_t t) { return (size_t) ((t >> 4) * 11400714819323198485ull) % TABLE; }

static void cb(MPI_T_event_instance ev, MPI_T_event_registration h __opal_attribute_unused__,
               MPI_T_cb_safety s __opal_attribute_unused__, void *user)
{
    int created = (int) (intptr_t) user;
    uint64_t token = 0;
    int32_t kind = -2;
    size_t i;
    (void) MPI_T_event_read(ev, 0, &token);   /* request, comm, kind, flags */
    (void) MPI_T_event_read(ev, 2, &kind);
    if (created) {
        ++n_created;
        if (kind < 0 || kind > 2) {
            ++n_bad_kind;
        }
        for (i = slot(token);; i = (i + 1) % TABLE) {
            if (0 == live[i] || 1 == live[i]) {
                live[i] = token;
                break;
            }
        }
    } else {
        ++n_freeing;
        for (i = slot(token); 0 != live[i]; i = (i + 1) % TABLE) {
            if (live[i] == token) {
                live[i] = 1;
                return;
            }
        }
        ++n_unmatched;                      /* a freeing event without a created event of the same token */
    }
}

int main(int argc, char **argv)
{
    int provided, size, rank, i_cr, i_fr, iters = argc > 1 ? atoi(argv[1]) : 20000;
    MPI_T_event_registration r_cr, r_fr;
    char s = 1, r = 0;
    long nb_expected = 0, k, still_live = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (MPI_SUCCESS != MPI_T_init_thread(MPI_THREAD_SINGLE, &provided)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_created", &i_cr)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_freeing", &i_fr)) {
        printf("SKIP: request handle events not available\n");
        MPI_Finalize();
        return 77;
    }
    /* not bound to an object: a tool cannot bind to a request that does not exist yet */
    expect("alloc handles", MPI_SUCCESS == MPI_T_event_handle_alloc(i_cr, NULL, MPI_INFO_NULL, &r_cr)
                                && MPI_SUCCESS == MPI_T_event_handle_alloc(i_fr, NULL, MPI_INFO_NULL, &r_fr));
    (void) MPI_T_event_register_callback(r_cr, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) 1, cb);
    (void) MPI_T_event_register_callback(r_fr, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) 0, cb);

    /* self messages: Isend + Irecv + Waitall on MPI_COMM_SELF, no partner needed */
    for (k = 0; k < iters; ++k) {
        MPI_Request rq[2];
        MPI_Irecv(&r, 1, MPI_CHAR, 0, 5, MPI_COMM_SELF, &rq[0]);
        MPI_Isend(&s, 1, MPI_CHAR, 0, 5, MPI_COMM_SELF, &rq[1]);
        MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
    }
    nb_expected += 2L * iters;
    /* A one-byte MPI_Isend usually completes inline and returns the shared pre-completed ("dummy") request: it has no
       created and no freeing event. Every other request has exactly one of each; the receives always do. */
    expect("p2p loop: created and freeing events match, at least one request per iteration (the receive)",
           n_created == n_freeing && n_created >= iters && n_created <= nb_expected);
    printf("INFO: p2p loop: %ld of %ld requests were real, %ld were dummy requests without events\n", n_created, nb_expected,
           nb_expected - n_created);

    /* persistent requests: created once, freeing once, restarted many times */
    {
        MPI_Request pr[2];
        long c0 = n_created, f0 = n_freeing;
        MPI_Recv_init(&r, 1, MPI_CHAR, 0, 6, MPI_COMM_SELF, &pr[0]);
        MPI_Send_init(&s, 1, MPI_CHAR, 0, 6, MPI_COMM_SELF, &pr[1]);
        for (k = 0; k < 1000; ++k) {
            MPI_Startall(2, pr);
            MPI_Waitall(2, pr, MPI_STATUSES_IGNORE);
        }
        expect("persistent: created once per request, not per start", n_created - c0 == 2 && n_freeing == f0);
        MPI_Request_free(&pr[0]);
        MPI_Request_free(&pr[1]);
        expect("persistent: freeing once per request", n_freeing - f0 == 2);
    }

    if (size >= 2) {                                           /* non-blocking collectives */
        long c0 = n_created, f0 = n_freeing;
        int buf = rank;
        for (k = 0; k < 1000; ++k) {
            MPI_Request rq;
            MPI_Ibcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD, &rq);
            MPI_Wait(&rq, MPI_STATUS_IGNORE);
        }
        expect("Ibcast loop: created and freeing events match (at least one request per call)",
               n_created - c0 >= 1000 && n_created - c0 == n_freeing - f0);
    }

    for (k = 0; k < TABLE; ++k) {
        if (live[k] > 1) {
            ++still_live;
        }
    }
    expect("every freeing event matches a created event of the same token", 0 == n_unmatched);
    expect("every created request was freed", 0 == still_live);
    expect("kinds are send, receive or collective", 0 == n_bad_kind);
    printf("INFO: rank %d: %ld created, %ld freeing\n", rank, n_created, n_freeing);

    (void) MPI_T_event_handle_free(r_cr, NULL, NULL);
    (void) MPI_T_event_handle_free(r_fr, NULL, NULL);
    MPI_T_finalize();
    MPI_Finalize();
    printf("RESULT: %s (%d failures)\n", 0 == failures ? "PASS" : "FAIL", failures);
    return 0 == failures ? 0 : 1;
}
