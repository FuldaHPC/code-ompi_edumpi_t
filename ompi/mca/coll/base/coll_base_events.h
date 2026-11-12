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
 * MPI_T event: which algorithm a collective component chose.
 *
 *   ompi.coll_algorithm_selected   bound to MPI_T_BIND_MPI_COMM
 *
 * It is raised once per collective operation by every component that picks an
 * algorithm (tuned, libnbc, han, ...), after the decision is made and before
 * the algorithm runs.  The algorithm is given as (component, operation,
 * algorithm id).  The id is the component's own number for the algorithm; for
 * tuned and libnbc it is the value of the component's
 * <component>_<operation>_algorithm control variable, whose enumeration gives
 * the name.  The communicator the event is bound to is the communicator the
 * collective runs on.
 *
 * A hierarchical component (han) runs further collectives on helper
 * communicators.  Those report themselves, with level 1 (inter-node) or 2
 * (intra-node) and parent_comm set to the user's communicator; the han
 * decision itself is reported with level 0.  The events of the helper
 * communicators are delivered to registrations on the user's communicator
 * (the communicator argument of the payload fields refers to the helper:
 * comm_size is the size of the helper).  Without han, level is 0 and
 * parent_comm is 0.
 */

#ifndef MCA_COLL_BASE_EVENTS_H
#define MCA_COLL_BASE_EVENTS_H

#include "ompi_config.h"

#include <assert.h>

#include "opal/mca/base/mca_base_event.h"
#include "ompi/communicator/communicator.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/coll/base/coll_base_functions.h"

BEGIN_C_DECLS

/* the component that made the decision */
enum {
    MCA_COLL_BASE_EVENT_COMPONENT_OTHER = 0,
    MCA_COLL_BASE_EVENT_COMPONENT_BASIC,
    MCA_COLL_BASE_EVENT_COMPONENT_TUNED,
    MCA_COLL_BASE_EVENT_COMPONENT_HAN,
    MCA_COLL_BASE_EVENT_COMPONENT_LIBNBC,
    MCA_COLL_BASE_EVENT_COMPONENT_ADAPT,
    MCA_COLL_BASE_EVENT_COMPONENT_XHC,
    MCA_COLL_BASE_EVENT_COMPONENT_ACOLL,
    MCA_COLL_BASE_EVENT_COMPONENT_UCC,
    MCA_COLL_BASE_EVENT_COMPONENT_SM,
    MCA_COLL_BASE_EVENT_COMPONENT_SELF,
    MCA_COLL_BASE_EVENT_COMPONENT_INTER,
};

/* level of the communicator the collective runs on */
enum {
    MCA_COLL_BASE_EVENT_LEVEL_USER = 0,
    MCA_COLL_BASE_EVENT_LEVEL_INTER_NODE = 1,
    MCA_COLL_BASE_EVENT_LEVEL_INTRA_NODE = 2,
};

struct mca_coll_base_algorithm_event_t {
    uint64_t request;     /* collective request (non-blocking operations), 0: blocking */
    int64_t msg_size;     /* bytes per process as the decision saw them, 0: none */
    uint64_t parent_comm; /* user communicator a hierarchical component works for, 0: none */
    int32_t operation;    /* COLLTYPE_T of coll_base_functions.h */
    int32_t algorithm;    /* component specific algorithm id */
    int32_t component;    /* MCA_COLL_BASE_EVENT_COMPONENT_* */
    int32_t comm_size;
    int32_t root;         /* -1: none */
    int32_t level;        /* MCA_COLL_BASE_EVENT_LEVEL_* */
};
typedef struct mca_coll_base_algorithm_event_t mca_coll_base_algorithm_event_t;

/* A hierarchical component (han) announces the helper communicators it runs
   collectives on, together with the user communicator they work for.  The raise
   helper looks the communicator up, only while the event is active. */
OMPI_DECLSPEC void mca_coll_base_event_helper_add(ompi_communicator_t *helper,
                                                  ompi_communicator_t *parent, int level);
OMPI_DECLSPEC void mca_coll_base_event_helper_remove(ompi_communicator_t *helper);
/* returns true and fills parent_comm/level if comm is a registered helper communicator */
OMPI_DECLSPEC bool mca_coll_base_event_helper_lookup(const ompi_communicator_t *comm,
                                                     uint64_t *parent_comm, int32_t *level);

OMPI_DECLSPEC extern mca_base_event_t *mca_coll_base_event_algorithm;

/* Called from the coll framework's register function. */
void mca_coll_base_events_register(void);

static inline bool mca_coll_base_event_algorithm_wanted(void)
{
    const mca_base_event_t *event = mca_coll_base_event_algorithm;

    return NULL != event && mca_base_event_active(event);
}

/* msg_bytes: bytes per process as the decision saw them */
static inline void mca_coll_base_event_raise_algorithm_bytes(ompi_communicator_t *comm,
                                                             int component, int operation,
                                                             int algorithm, size_t msg_bytes,
                                                             int root, const void *request)
{
    uint64_t parent = 0;
    int32_t level = MCA_COLL_BASE_EVENT_LEVEL_USER;

    if (OPAL_LIKELY(!mca_coll_base_event_algorithm_wanted())) {
        return;
    }
    (void) mca_coll_base_event_helper_lookup(comm, &parent, &level);

    mca_coll_base_algorithm_event_t payload = {.request = (uint64_t) (uintptr_t) request,
                                               .msg_size = (int64_t) msg_bytes,
                                               .parent_comm = parent,
                                               .operation = operation,
                                               .algorithm = algorithm,
                                               .component = component,
                                               .comm_size = ompi_comm_size(comm),
                                               .root = root,
                                               .level = level};
    /* The event of a helper communicator is delivered to registrations on the
       user communicator it works for: a tool cannot see the helper. */
    mca_base_event_raise_bound(mca_coll_base_event_algorithm, NULL,
                               0 != parent ? (ompi_communicator_t *) (uintptr_t) parent : comm,
                               &payload);
}

/* msg size = count * size of dtype (0 if dtype is NULL) */
static inline void mca_coll_base_event_raise_algorithm(ompi_communicator_t *comm, int component,
                                                       int operation, int algorithm, size_t count,
                                                       ompi_datatype_t *dtype, int root,
                                                       const void *request)
{
    size_t size = 0;

    if (OPAL_LIKELY(!mca_coll_base_event_algorithm_wanted())) {
        return;
    }
    if (NULL != dtype) {
        ompi_datatype_type_size(dtype, &size);
    }
    mca_coll_base_event_raise_algorithm_bytes(comm, component, operation, algorithm, size * count,
                                              root, request);
}

END_C_DECLS

#endif /* MCA_COLL_BASE_EVENTS_H */
