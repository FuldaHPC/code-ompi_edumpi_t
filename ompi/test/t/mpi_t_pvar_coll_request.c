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
 * Request-bound PVARs of non-blocking collectives (the scenario of the MPI Forum Tools WG side document): the handles of
 * coll_libnbc_request_messages and coll_libnbc_request_bytes are allocated in the handler of ompi.mpi.request_created
 * for the requests of two overlapping MPI_Iallreduce on a duplicate of MPI_COMM_WORLD (with a blocking MPI_Allreduce
 * and a ring of MPI_Sendrecv in between) and read in the handler of ompi.mpi.request_freeing.  The per-peer counts must
 * equal the counts obtained independently from the ompi.pml_send_start events whose parent element names the request.
 * Needs at least two processes (skips with one).
 */

#include "opal_config.h"

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXREQ 8
#define MAXPEER 64

static int failures = 0;
static volatile int track = 0;    /* only the two requests created by this test (HAN creates more on the same communicator) */
static MPI_Comm g_comm;
static uint64_t g_comm_handle;
static MPI_T_pvar_session session;
static int idx_msgs = -1, idx_bytes = -1, g_size = 0;

typedef struct {
    uint64_t token;
    MPI_T_pvar_handle h_msgs, h_bytes;
    unsigned long long ev_msgs[MAXPEER], ev_bytes[MAXPEER];   /* from the send_start events with this parent */
    int bound, done, equal, total;
} req_t;
static req_t reqs[MAXREQ];
static int nreq = 0;

static void expect(const char *what, int cond)
{
    printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) {
        ++failures;
    }
}

static req_t *find(uint64_t token)
{
    for (int i = 0; i < nreq; ++i) {
        if (reqs[i].token == token) {
            return &reqs[i];
        }
    }
    return NULL;
}

static void cb_handle(MPI_T_event_instance ev, MPI_T_event_registration h __opal_attribute_unused__,
                      MPI_T_cb_safety s __opal_attribute_unused__, void *user)
{
    int created = (int) (intptr_t) user, cnt;
    uint64_t token = 0, comm = 0;
    int32_t kind = -1;
    (void) MPI_T_event_read(ev, 0, &token);   /* request, comm, kind, flags */
    (void) MPI_T_event_read(ev, 1, &comm);
    (void) MPI_T_event_read(ev, 2, &kind);
    if (2 != kind || comm != g_comm_handle) {
        return;                              /* requests of other communicators (HAN's internal ones) */
    }
    if (created) {
        MPI_Request rq = (MPI_Request) (uintptr_t) token;
        if (!track) {
            return;
        }
        req_t *r = (nreq < MAXREQ) ? &reqs[nreq++] : NULL;
        if (r) {
            memset(r, 0, sizeof(*r));
            r->token = token;
            r->bound = (MPI_SUCCESS == MPI_T_pvar_handle_alloc(session, idx_msgs, &rq, &r->h_msgs, &cnt) && cnt == g_size
                        && MPI_SUCCESS == MPI_T_pvar_handle_alloc(session, idx_bytes, &rq, &r->h_bytes, &cnt) && cnt == g_size);
        }
    } else {
        req_t *r = find(token);
        if (r && r->bound) {
            unsigned long long m[MAXPEER], b[MAXPEER];
            int ok = (MPI_SUCCESS == MPI_T_pvar_read(session, r->h_msgs, m) && MPI_SUCCESS == MPI_T_pvar_read(session, r->h_bytes, b));
            r->equal = ok;
            r->total = 0;
            for (int i = 0; i < g_size && ok; ++i) {
                r->total += (int) m[i];
                if (m[i] != r->ev_msgs[i] || b[i] != r->ev_bytes[i]) {
                    r->equal = 0;
                }
            }
            r->done = 1;
            (void) MPI_T_pvar_handle_free(session, &r->h_msgs);
            (void) MPI_T_pvar_handle_free(session, &r->h_bytes);
        }
    }
}

static void cb_send(MPI_T_event_instance ev, MPI_T_event_registration h __opal_attribute_unused__,
                    MPI_T_cb_safety s __opal_attribute_unused__, void *user __opal_attribute_unused__)
{
    uint64_t parent = 0;
    int64_t len = 0;
    int32_t peer = -1;
    /* request, parent, length, peer, peer_world, tag, flags */
    (void) MPI_T_event_read(ev, 1, &parent);
    if (0 == parent) {
        return;
    }
    (void) MPI_T_event_read(ev, 2, &len);
    (void) MPI_T_event_read(ev, 3, &peer);
    req_t *r = find(parent);
    if (r && peer >= 0 && peer < MAXPEER) {
        r->ev_msgs[peer] += 1;
        r->ev_bytes[peer] += (unsigned long long) len;
    }
}

int main(int argc, char **argv)
{
    int provided, size, rank, i_cr, i_fr, i_sd, cls_ok = 1;
    MPI_T_event_registration r_cr, r_fr, r_sd;
    MPI_Request rq[2];
    double a[10] = {0}, ar[10], b[100] = {0}, br[100], c[10] = {0}, cr[10];
    char sbuf[64] = {0}, rbuf[64];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2) {
        printf("SKIP: needs at least 2 processes\n");
        MPI_Finalize();
        return 77;
    }
    g_size = size;
    if (MPI_SUCCESS != MPI_T_init_thread(MPI_THREAD_SINGLE, &provided)
        || MPI_SUCCESS != MPI_T_pvar_get_index("coll_libnbc_request_messages", MPI_T_PVAR_CLASS_COUNTER, &idx_msgs)
        || MPI_SUCCESS != MPI_T_pvar_get_index("coll_libnbc_request_bytes", MPI_T_PVAR_CLASS_COUNTER, &idx_bytes)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_created", &i_cr)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.mpi.request_freeing", &i_fr)
        || MPI_SUCCESS != MPI_T_event_get_index("ompi.pml_send_start", &i_sd)) {
        printf("SKIP: request-bound variables or request events not available\n");
        MPI_Finalize();
        return 77;
    }
    (void) cls_ok;
    MPI_Comm_dup(MPI_COMM_WORLD, &g_comm);
    g_comm_handle = (uint64_t) (uintptr_t) g_comm;
    expect("session", MPI_SUCCESS == MPI_T_pvar_session_create(&session));
    expect("event handles", MPI_SUCCESS == MPI_T_event_handle_alloc(i_cr, NULL, MPI_INFO_NULL, &r_cr)
                                && MPI_SUCCESS == MPI_T_event_handle_alloc(i_fr, NULL, MPI_INFO_NULL, &r_fr)
                                && MPI_SUCCESS == MPI_T_event_handle_alloc(i_sd, &g_comm, MPI_INFO_NULL, &r_sd));
    (void) MPI_T_event_register_callback(r_cr, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) 1, cb_handle);
    (void) MPI_T_event_register_callback(r_fr, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, (void *) 0, cb_handle);
    (void) MPI_T_event_register_callback(r_sd, MPI_T_CB_REQUIRE_NONE, MPI_INFO_NULL, NULL, cb_send);

    track = 1;
    MPI_Iallreduce(a, ar, 10, MPI_DOUBLE, MPI_SUM, g_comm, &rq[0]);
    MPI_Iallreduce(b, br, 100, MPI_DOUBLE, MPI_SUM, g_comm, &rq[1]);
    track = 0;
    MPI_Allreduce(c, cr, 10, MPI_DOUBLE, MPI_SUM, g_comm);
    MPI_Sendrecv(sbuf, 64, MPI_CHAR, (rank + 1) % size, 3, rbuf, 64, MPI_CHAR, (rank + size - 1) % size, 3, g_comm, MPI_STATUS_IGNORE);
    MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);

    expect("two collective requests were seen on the communicator", 2 == nreq);
    for (int i = 0; i < nreq; ++i) {
        char what[160];
        snprintf(what, sizeof(what), "request %d: handles bound in the created handler, read in the freeing handler", i);
        expect(what, reqs[i].bound && reqs[i].done);
        snprintf(what, sizeof(what), "request %d: per-peer messages and bytes equal the counts from the send_start events with this parent (%d messages)",
                 i, reqs[i].total);
        expect(what, reqs[i].equal && reqs[i].total > 0);
    }
    if (2 == nreq) {
        unsigned long long b0 = 0, b1 = 0;
        for (int i = 0; i < size; ++i) {
            b0 += reqs[0].ev_bytes[i];
            b1 += reqs[1].ev_bytes[i];
        }
        expect("the 800 byte request moved more than the 80 byte request: nothing leaked between them", b1 > b0 && b0 > 0);
    }
    /* refuse a request that is not a libnbc request */
    {
        MPI_Request pr;
        MPI_T_pvar_handle h;
        int cnt;
        char x = 0;
        MPI_Irecv(&x, 1, MPI_CHAR, MPI_PROC_NULL, 0, MPI_COMM_SELF, &pr);
        expect("binding to a point-to-point request is refused", MPI_SUCCESS != MPI_T_pvar_handle_alloc(session, idx_msgs, &pr, &h, &cnt));
        MPI_Wait(&pr, MPI_STATUS_IGNORE);
    }

    (void) MPI_T_event_handle_free(r_cr, NULL, NULL);
    (void) MPI_T_event_handle_free(r_fr, NULL, NULL);
    (void) MPI_T_event_handle_free(r_sd, NULL, NULL);
    MPI_Comm_free(&g_comm);
    MPI_T_finalize();
    MPI_Finalize();
    printf("RESULT: %s (%d failures)\n", 0 == failures ? "PASS" : "FAIL", failures);
    return 0 == failures ? 0 : 1;
}
