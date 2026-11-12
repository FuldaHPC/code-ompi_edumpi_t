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
 * Generic request state ompi.mpi.request_cancelled (REQUEST_CANCELLED of the MPI Forum Tools WG side document): a
 * receive that never matches is cancelled; the tool must see request_created, request_cancelled and request_freeing
 * with the same token, in this order, and the status must say cancelled.  Runs with one process.
 */

#include "opal_config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>

static int failures = 0;
static int n_created = 0, n_cancelled = 0, n_freeing = 0, order = 0, order_ok = 1;
static uint64_t t_created = 0, t_cancelled = 0, t_freeing = 0;
static int32_t kind_cancelled = -1;

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
    int which = (int) (intptr_t) user;       /* 0 created, 1 cancelled, 2 freeing */
    uint64_t token = 0;
    int32_t kind = -1;
    (void) MPI_T_event_read(ev, 0, &token);
    (void) MPI_T_event_read(ev, 2, &kind);
    if (0 == which) {
        ++n_created; t_created = token; order_ok &= (0 == order); order = 1;
    } else if (1 == which) {
        ++n_cancelled; t_cancelled = token; kind_cancelled = kind; order_ok &= (1 == order); order = 2;
    } else {
        ++n_freeing; t_freeing = token; order_ok &= (2 == order); order = 3;
    }
}

int main(int argc, char **argv)
{
    int provided, i_cr, i_ca, i_fr, flag = 0;
    MPI_T_event_registration r[3];
    MPI_Comm self = MPI_COMM_SELF;
    MPI_Request rq;
    MPI_Status st;
    char c = 0;

    MPI_Init(&argc, &argv);
    if (MPI_SUCCESS != MPI_T_init_thread(MPI_THREAD_SINGLE, &provided)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_created", &i_cr)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_cancelled", &i_ca)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_freeing", &i_fr)) {
        printf("SKIP: request cancelled event not available\n");
        MPI_Finalize();
        return 77;
    }
    /* created and freeing are not bound; cancelled is bound to the communicator of the request */
    expect("alloc handles", MPI_SUCCESS == MPI_T_event_handle_alloc(i_cr, NULL, MPI_INFO_NULL, &r[0])
                                && MPI_SUCCESS == MPI_T_event_handle_alloc(i_ca, &self, MPI_INFO_NULL, &r[1])
                                && MPI_SUCCESS == MPI_T_event_handle_alloc(i_fr, NULL, MPI_INFO_NULL, &r[2]));
    for (int i = 0; i < 3; ++i) {
        (void) MPI_T_event_register_callback(r[i], MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) (intptr_t) i, cb);
    }

    MPI_Irecv(&c, 1, MPI_CHAR, 0, 77, self, &rq);      /* nobody sends tag 77 */
    MPI_Cancel(&rq);
    MPI_Wait(&rq, &st);
    MPI_Test_cancelled(&st, &flag);
    expect("the receive was cancelled (status)", flag);
    expect("one created, one cancelled, one freeing event", 1 == n_created && 1 == n_cancelled && 1 == n_freeing);
    expect("same token in all three", t_created != 0 && t_created == t_cancelled && t_created == t_freeing);
    expect("order: created, cancelled, freeing", order_ok && 3 == order);
    expect("kind of the cancelled event is a receive (1)", 1 == kind_cancelled);

    for (int i = 0; i < 3; ++i) {
        (void) MPI_T_event_handle_free(r[i], NULL, NULL);
    }
    MPI_T_finalize();
    MPI_Finalize();
    printf("RESULT: %s (%d failures)\n", 0 == failures ? "PASS" : "FAIL", failures);
    return 0 == failures ? 0 : 1;
}
