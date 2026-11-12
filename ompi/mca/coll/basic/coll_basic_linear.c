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
 * The basic component uses the linear algorithms of the coll base library for
 * most intra-communicator operations.  These thin wrappers report the choice
 * through the MPI_T event ompi.coll_algorithm_selected (algorithm 1: linear)
 * and then call the base algorithm.
 */

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_base_events.h"
#include "coll_basic.h"

int mca_coll_basic_barrier_intra_linear(struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, BARRIER, 1,
                                        0, NULL, -1, NULL);
    return ompi_coll_base_barrier_intra_basic_linear(comm, module);
}

int mca_coll_basic_bcast_intra_linear(void *buffer, size_t count, struct ompi_datatype_t *datatype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, BCAST, 1,
                                        count, datatype, root, NULL);
    return ompi_coll_base_bcast_intra_basic_linear(buffer, count, datatype, root, comm, module);
}

int mca_coll_basic_reduce_intra_linear(const void *sendbuf, void *recvbuf, size_t count, struct ompi_datatype_t *datatype, struct ompi_op_t *op, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, REDUCE, 1,
                                        count, datatype, root, NULL);
    return ompi_coll_base_reduce_intra_basic_linear(sendbuf, recvbuf, count, datatype, op, root, comm, module);
}

int mca_coll_basic_allgather_intra_linear(const void *sendbuf, size_t sendcount, struct ompi_datatype_t *sendtype, void *recvbuf, size_t recvcount, struct ompi_datatype_t *recvtype, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, ALLGATHER, 1,
                                        sendcount, sendtype, -1, NULL);
    return ompi_coll_base_allgather_intra_basic_linear(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, module);
}

int mca_coll_basic_allgatherv_intra_linear(const void *sendbuf, size_t sendcount, struct ompi_datatype_t *sendtype, void *recvbuf, ompi_count_array_t recvcounts, ompi_disp_array_t displs, struct ompi_datatype_t *recvtype, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, ALLGATHERV, 1,
                                        sendcount, sendtype, -1, NULL);
    return ompi_coll_base_allgatherv_intra_basic_default(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, module);
}

int mca_coll_basic_alltoall_intra_linear(const void *sendbuf, size_t sendcount, struct ompi_datatype_t *sendtype, void *recvbuf, size_t recvcount, struct ompi_datatype_t *recvtype, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, ALLTOALL, 1,
                                        sendcount, sendtype, -1, NULL);
    return ompi_coll_base_alltoall_intra_basic_linear(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, module);
}

int mca_coll_basic_alltoallv_intra_linear(const void *sendbuf, ompi_count_array_t sendcounts, ompi_disp_array_t sdispls, struct ompi_datatype_t *sendtype, void *recvbuf, ompi_count_array_t recvcounts, ompi_disp_array_t rdispls, struct ompi_datatype_t *recvtype, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, ALLTOALLV, 1,
                                        0, sendtype, -1, NULL);
    return ompi_coll_base_alltoallv_intra_basic_linear(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, module);
}

int mca_coll_basic_exscan_intra_linear(const void *sendbuf, void *recvbuf, size_t count, struct ompi_datatype_t *datatype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, EXSCAN, 1,
                                        count, datatype, -1, NULL);
    return ompi_coll_base_exscan_intra_linear(sendbuf, recvbuf, count, datatype, op, comm, module);
}

int mca_coll_basic_gather_intra_linear(const void *sendbuf, size_t sendcount, struct ompi_datatype_t *sendtype, void *recvbuf, size_t recvcount, struct ompi_datatype_t *recvtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, GATHER, 1,
                                        sendcount, sendtype, root, NULL);
    return ompi_coll_base_gather_intra_basic_linear(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, module);
}

int mca_coll_basic_scan_intra_linear(const void *sendbuf, void *recvbuf, size_t count, struct ompi_datatype_t *datatype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, SCAN, 1,
                                        count, datatype, -1, NULL);
    return ompi_coll_base_scan_intra_linear(sendbuf, recvbuf, count, datatype, op, comm, module);
}

int mca_coll_basic_scatter_intra_linear(const void *sendbuf, size_t sendcount, struct ompi_datatype_t *sendtype, void *recvbuf, size_t recvcount, struct ompi_datatype_t *recvtype, int root, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    mca_coll_base_event_raise_algorithm(comm, MCA_COLL_BASE_EVENT_COMPONENT_BASIC, SCATTER, 1,
                                        sendcount, sendtype, root, NULL);
    return ompi_coll_base_scatter_intra_basic_linear(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, module);
}

