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
 * Coverage for the generic request state events ompi.mpi.request_activated and ompi.mpi.request_completed and for the
 * "parent" element of ompi.pml_send_start.  Needs at least two processes (skips with one).  A non-blocking collective
 * on a duplicate of MPI_COMM_WORLD must raise exactly one activated and one completed event with kind 2 (collective)
 * and the same token, the completed event after the activated one; every internal message (negative tag) that the
 * schedule sends must name that token as its parent.
 */

#include "opal_config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXEV 256
static int failures = 0;
static int n_act = 0, n_comp = 0, n_internal = 0, n_with_parent = 0, order_ok = 1;
static uint64_t act_token = 0, comp_token = 0, wrong_parent = 0;
static int act_kind = -1, comp_kind = -1;

static void expect(const char *what, int cond)
{
    printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) {
        ++failures;
    }
}

static void cb_state(MPI_T_event_instance ev, MPI_T_event_registration h __opal_attribute_unused__,
                     MPI_T_cb_safety s __opal_attribute_unused__, void *user)
{
    int activated = (int) (intptr_t) user;
    uint64_t token = 0;
    int32_t kind = -1;
    (void) MPI_T_event_read(ev, 0, &token);   /* request, comm, kind, flags */
    (void) MPI_T_event_read(ev, 2, &kind);
    if (2 != kind) {
        return;
    }
    if (activated) {
        ++n_act;
        act_token = token;
        act_kind = kind;
    } else {
        ++n_comp;
        comp_token = token;
        comp_kind = kind;
        if (0 == n_act) {
            order_ok = 0;
        }
    }
}

static void cb_send(MPI_T_event_instance ev, MPI_T_event_registration h __opal_attribute_unused__,
                    MPI_T_cb_safety s __opal_attribute_unused__, void *user __opal_attribute_unused__)
{
    uint64_t parent = 0;
    int32_t tag = 0;
    /* request, parent, length, peer, peer_world, tag, flags */
    (void) MPI_T_event_read(ev, 1, &parent);
    (void) MPI_T_event_read(ev, 5, &tag);
    if (tag < 0) {
        ++n_internal;
        if (parent) {
            ++n_with_parent;
            if (parent != act_token) {
                wrong_parent = parent;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int provided, size, i_act, i_comp, i_send;
    MPI_Comm comm;
    MPI_T_event_registration r_act, r_comp, r_send;
    MPI_Request rq;
    int buf[1024] = {0};

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        printf("SKIP: needs at least 2 processes\n");
        MPI_Finalize();
        return 77;
    }
    if (MPI_SUCCESS != MPI_T_init_thread(MPI_THREAD_SINGLE, &provided)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_activated", &i_act)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_completed", &i_comp)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.pml_send_start", &i_send)) {
        printf("SKIP: request state events not available\n");
        MPI_Finalize();
        return 77;
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    expect("alloc handles", MPI_SUCCESS == MPI_T_event_handle_alloc(i_act, &comm, MPI_INFO_NULL, &r_act)
                                && MPI_SUCCESS == MPI_T_event_handle_alloc(i_comp, &comm, MPI_INFO_NULL, &r_comp)
                                && MPI_SUCCESS == MPI_T_event_handle_alloc(i_send, &comm, MPI_INFO_NULL, &r_send));
    (void) MPI_T_event_register_callback(r_act, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) 1, cb_state);
    (void) MPI_T_event_register_callback(r_comp, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) 0, cb_state);
    (void) MPI_T_event_register_callback(r_send, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, NULL, cb_send);

    MPI_Ibcast(buf, 1024, MPI_INT, 0, comm, &rq);
    MPI_Wait(&rq, MPI_STATUS_IGNORE);

    expect("one REQUEST_ACTIVATED of kind collective", 1 == n_act && 2 == act_kind);
    expect("one REQUEST_COMPLETED of kind collective", 1 == n_comp && 2 == comp_kind);
    expect("same token in both events", act_token != 0 && act_token == comp_token);
    expect("completed after activated", order_ok);
    expect("the token is the MPI_Request handle", act_token == (uint64_t) (uintptr_t) rq || MPI_REQUEST_NULL == rq);
    printf("INFO: rank %d: %d internal sends, %d with a parent\n", 0, n_internal, n_with_parent);
    expect("every internal send of the schedule names the request as parent (if this rank sent any)",
           0 == wrong_parent && n_internal == n_with_parent);

    (void) MPI_T_event_handle_free(r_act, NULL, NULL);
    (void) MPI_T_event_handle_free(r_comp, NULL, NULL);
    (void) MPI_T_event_handle_free(r_send, NULL, NULL);
    MPI_Comm_free(&comm);
    MPI_T_finalize();
    MPI_Finalize();
    printf("RESULT: %s (%d failures)\n", 0 == failures ? "PASS" : "FAIL", failures);
    return 0 == failures ? 0 : 1;
}
