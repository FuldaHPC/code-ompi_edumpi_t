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

#include "ompi/mca/pml/base/pml_base_events.h"
#include "opal/mca/base/mca_base_pvar.h"
#include "opal/mca/base/mca_base_var.h"

mca_base_event_t *mca_pml_base_event[MCA_PML_BASE_EVENT_MAX];

/* Nanosecond ticks: the same "ompi" source the ob1 events and the MPI layer use
   (registration by name is idempotent). */
#define PML_BASE_TICKS_PER_SECOND ((opal_count_t) 1000000000)

static const mca_base_var_type_t start_types[] = {
    MCA_BASE_VAR_TYPE_UINT64_T, MCA_BASE_VAR_TYPE_UINT64_T, MCA_BASE_VAR_TYPE_INT64_T,
    MCA_BASE_VAR_TYPE_INT32_T,  MCA_BASE_VAR_TYPE_INT32_T,  MCA_BASE_VAR_TYPE_INT32_T,
    MCA_BASE_VAR_TYPE_INT32_T};
static const ptrdiff_t start_offsets[] = {
    offsetof(mca_pml_base_start_event_t, request),    offsetof(mca_pml_base_start_event_t, parent),
    offsetof(mca_pml_base_start_event_t, length),     offsetof(mca_pml_base_start_event_t, peer),
    offsetof(mca_pml_base_start_event_t, peer_world), offsetof(mca_pml_base_start_event_t, tag),
    offsetof(mca_pml_base_start_event_t, flags)};

static const mca_base_var_type_t complete_types[] = {
    MCA_BASE_VAR_TYPE_UINT64_T, MCA_BASE_VAR_TYPE_INT64_T, MCA_BASE_VAR_TYPE_INT32_T,
    MCA_BASE_VAR_TYPE_INT32_T,  MCA_BASE_VAR_TYPE_INT32_T, MCA_BASE_VAR_TYPE_INT32_T};
static const ptrdiff_t complete_offsets[] = {
    offsetof(mca_pml_base_complete_event_t, request),
    offsetof(mca_pml_base_complete_event_t, length),
    offsetof(mca_pml_base_complete_event_t, peer),
    offsetof(mca_pml_base_complete_event_t, peer_world),
    offsetof(mca_pml_base_complete_event_t, tag),
    offsetof(mca_pml_base_complete_event_t, flags)};

struct pml_base_event_desc_t {
    int which;
    const char *name;
    const char *description;
    int num_elements;
    const mca_base_var_type_t *types;
    const ptrdiff_t *offsets;
};

#define PML_BASE_START_PAYLOAD 7, start_types, start_offsets
#define PML_BASE_COMPLETE_PAYLOAD 6, complete_types, complete_offsets

static const struct pml_base_event_desc_t pml_base_events[] = {
    {MCA_PML_BASE_EVENT_SEND_START, "send_start",
     "A send was issued: destination, tag, length in bytes and mode flags. "
     "Independent of the PML in use. A negative tag marks an internal message of a collective.",
     PML_BASE_START_PAYLOAD},
    {MCA_PML_BASE_EVENT_RECV_START, "recv_start",
     "A receive was posted: source (or MPI_ANY_SOURCE), tag and receive buffer capacity in bytes. "
     "Independent of the PML in use.",
     PML_BASE_START_PAYLOAD},
    {MCA_PML_BASE_EVENT_SEND_COMPLETE, "send_complete",
     "A send completed locally: destination, tag and bytes sent. Independent of the PML in use.",
     PML_BASE_COMPLETE_PAYLOAD},
    {MCA_PML_BASE_EVENT_RECV_COMPLETE, "recv_complete",
     "A receive completed: actual source, actual tag and bytes received. Independent of the "
     "PML in use.",
     PML_BASE_COMPLETE_PAYLOAD},
};

void mca_pml_base_events_register(void)
{
    /* The master switch over Open MPI's built-in producers belongs to the MPI
       layer (ompi_mpit_register_events.c); we only read it. */
    const int *register_producers = NULL;
    mca_base_event_source_t *source = NULL;
    int source_index, var_index;
    size_t i;

    var_index = mca_base_var_find("opal", "mca", "base", "event_register_producers");
    if (0 <= var_index) {
        (void) mca_base_var_get_value(var_index, &register_producers, NULL, NULL);
    }

    for (i = 0; i < MCA_PML_BASE_EVENT_MAX; ++i) {
        mca_pml_base_event[i] = NULL;
    }

    if (NULL != register_producers && 0 == register_producers[0]) {
        return;
    }

    source_index = mca_base_event_source_register("ompi", "Open MPI runtime events (ordered)",
                                                  MCA_BASE_EVENT_SOURCE_ORDERED,
                                                  PML_BASE_TICKS_PER_SECOND, OPAL_COUNT_MAX, NULL,
                                                  true, NULL);
    if (0 > source_index
        || OPAL_SUCCESS != mca_base_event_source_get_by_index(source_index, &source)) {
        return;
    }

    for (i = 0; i < sizeof(pml_base_events) / sizeof(pml_base_events[0]); ++i) {
        const struct pml_base_event_desc_t *desc = &pml_base_events[i];
        int event_index;

        /* framework-level event: full name is "ompi.pml_<name>" */
        event_index = mca_base_event_register("ompi", "pml", NULL, desc->name, desc->description,
                                              OPAL_INFO_LVL_4, desc->num_elements, desc->types,
                                              desc->offsets, NULL, MCA_BASE_VAR_BIND_MPI_COMM,
                                              MCA_BASE_EVENT_FLAG_IWG, source, NULL);
        if (0 > event_index) {
            continue;
        }
        (void) mca_base_event_get_by_index(event_index, &mca_pml_base_event[desc->which]);
    }
}
