/*
 * Copyright (c) 2021-2026 Computer Architecture and VLSI Systems (CARV)
 *                         Laboratory, ICS Forth. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 * SPDX-License-Identifier: BSD-3-Clause-Open-MPI
 */

#include "ompi_config.h"
#include "ompi/mca/coll/base/coll_base_events.h"
#include "mpi.h"

#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/op/op.h"

#include "coll_xhc.h"

int mca_coll_xhc_reduce(const void *sbuf, void *rbuf,
        size_t count, ompi_datatype_t *datatype, ompi_op_t *op, int root,
        ompi_communicator_t *ompi_comm, mca_coll_base_module_t *ompi_module) {
    mca_coll_base_event_raise_algorithm(ompi_comm, MCA_COLL_BASE_EVENT_COMPONENT_XHC, REDUCE, 1,
                                        count, datatype, root, NULL);

    return xhc_allreduce_internal(sbuf, rbuf, count, datatype,
        op, root, ompi_comm, ompi_module, XHC_REDUCE);
}
