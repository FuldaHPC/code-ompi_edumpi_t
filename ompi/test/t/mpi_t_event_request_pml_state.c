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
 * Coverage for ompi.mpi.request_activated / request_completed raised for point-to-point requests.  A non-blocking
 * self send and receive must each give one activated and one completed event, with kind 0 (send) / 1 (receive), the
 * token equal to the MPI_Request handle, activated before completed.  A persistent send started twice gives two
 * pairs with the PERSISTENT flag.
 */

#include "opal_config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>

#define MAXT 8
static int failures = 0;
static struct { uint64_t token; int kind, flags, act, comp, order_ok; } tab[MAXT];
static int ntab = 0;

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
    int activated = (int) (intptr_t) user;
    uint64_t token = 0;
    int32_t kind = -1, flags = 0;
    (void) MPI_T_event_read(ev, 0, &token);
    (void) MPI_T_event_read(ev, 2, &kind);
    (void) MPI_T_event_read(ev, 3, &flags);
    if (kind > 1) {
        return;
    }
    int i = 0;
    for (; i < ntab; ++i) {
        if (tab[i].token == token) {
            break;
        }
    }
    if (i == ntab) {
        if (ntab == MAXT) {
            return;
        }
        tab[ntab].token = token;
        tab[ntab].kind = kind;
        tab[ntab].order_ok = 1;
        ++ntab;
    }
    if (activated) {
        ++tab[i].act;
        tab[i].flags |= flags;
    } else {
        ++tab[i].comp;
        if (tab[i].comp > tab[i].act) {
            tab[i].order_ok = 0;
        }
    }
}

static int find(MPI_Request rq)
{
    for (int i = 0; i < ntab; ++i) {
        if (tab[i].token == (uint64_t) (uintptr_t) rq) {
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv)
{
    int provided, i_act, i_comp, in = 7, out = 7;
    MPI_T_event_registration r[2];
    MPI_Comm self = MPI_COMM_SELF;
    MPI_Request rs, rr, rp;

    MPI_Init(&argc, &argv);
    if (MPI_SUCCESS != MPI_T_init_thread(MPI_THREAD_SINGLE, &provided)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_activated", &i_act)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_completed", &i_comp)) {
        printf("SKIP: generic request state events not available\n");
        MPI_Finalize();
        return 77;
    }
    expect("alloc handles", MPI_SUCCESS == MPI_T_event_handle_alloc(i_act, &self, MPI_INFO_NULL, &r[0])
                                && MPI_SUCCESS == MPI_T_event_handle_alloc(i_comp, &self, MPI_INFO_NULL, &r[1]));
    (void) MPI_T_event_register_callback(r[0], MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) (intptr_t) 1, cb);
    (void) MPI_T_event_register_callback(r[1], MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) (intptr_t) 0, cb);

    MPI_Irecv(&in, 1, MPI_INT, 0, 5, self, &rr);
    MPI_Isend(&out, 1, MPI_INT, 0, 5, self, &rs);
    uint64_t trr = (uint64_t) (uintptr_t) rr, trs = (uint64_t) (uintptr_t) rs;
    MPI_Wait(&rs, MPI_STATUS_IGNORE);
    MPI_Wait(&rr, MPI_STATUS_IGNORE);
    int ir = -1, is = -1;
    for (int i = 0; i < ntab; ++i) {
        if (tab[i].token == trr) ir = i;
        if (tab[i].token == trs) is = i;
    }
    expect("receive: one activated, one completed, kind 1", ir >= 0 && 1 == tab[ir].act && 1 == tab[ir].comp
                                                              && 1 == tab[ir].kind && tab[ir].order_ok);
    expect("send: one activated, one completed, kind 0", is >= 0 && 1 == tab[is].act && 1 == tab[is].comp
                                                            && 0 == tab[is].kind && tab[is].order_ok);

    ntab = 0;
    MPI_Send_init(&out, 1, MPI_INT, 0, 6, self, &rp);
    for (int k = 0; k < 2; ++k) {
        MPI_Irecv(&in, 1, MPI_INT, 0, 6, self, &rr);
        MPI_Start(&rp);
        MPI_Wait(&rp, MPI_STATUS_IGNORE);
        MPI_Wait(&rr, MPI_STATUS_IGNORE);
    }
    int ip = find(rp);
    expect("persistent send: two activated, two completed, PERSISTENT flag",
           ip >= 0 && 2 == tab[ip].act && 2 == tab[ip].comp && (tab[ip].flags & 1) && tab[ip].order_ok);
    MPI_Request_free(&rp);

    for (int i = 0; i < 2; ++i) {
        (void) MPI_T_event_handle_free(r[i], NULL, NULL);
    }
    MPI_T_finalize();
    MPI_Finalize();
    printf("RESULT: %s (%d failures)\n", 0 == failures ? "PASS" : "FAIL", failures);
    return 0 == failures ? 0 : 1;
}
