# coding: utf-8

import itertools
import numpy as np
from itertools import product
from mpi4py    import MPI

from psydac.ddm.partition import compute_dims, partition_procs_per_patch


__all__ = ['find_mpi_type', 'CartDecomposition', 'CartDataExchanger']

#===============================================================================
def find_mpi_type( dtype ):
    """
    Find correct MPI datatype that corresponds to user-provided datatype.

    Parameters
    ----------
    dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
        Datatype for which the corresponding MPI datatype is requested.

    Returns
    -------
    mpi_type : mpi4py.MPI.Datatype
        MPI datatype to be used for communication.

    """
    if isinstance( dtype, MPI.Datatype ):
        mpi_type = dtype
    else:
        nt = np.dtype( dtype )
        mpi_type = MPI._typedict[nt.char]

    return mpi_type

#====================================================================================
class MultiCartDecomposition():
    def __init__( self, npts, pads, periods, reorder, comm=MPI.COMM_WORLD, shifts=None, num_threads=None ):

        # Check input arguments
        # TODO: check that arguments are identical across all processes
        assert len( npts ) == len( pads ) == len( periods )
        assert all(all( n >=1 for n in npts_k ) for npts_k in npts)
        assert isinstance( comm, MPI.Comm )

        shifts      = tuple(shifts) if shifts else [(1,)*len(npts[0]) for n in npts]
        num_threads = num_threads if num_threads else 1

        # Store input arguments
        self._npts         = tuple( npts    )
        self._pads         = tuple( pads    )
        self._periods      = tuple( periods )
        self._shifts       = tuple( shifts  )
        self._num_threads  = num_threads
        self._comm         = comm

        # ...
        self._ncarts = len( npts )

        # ...
        size           = comm.Get_size()
        rank           = comm.Get_rank()
        sizes, rank_ranges = partition_procs_per_patch(self._npts, size)

        self._rank   = rank
        self._size   = size
        self._sizes  = sizes
        self._rank_ranges = rank_ranges

        global_group = comm.group
        owned_groups = []

        local_groups        = [None]*self._ncarts
        local_communicators = [None]*self._ncarts

        for i,r in enumerate(rank_ranges):
            if rank>=r[0] and rank<=r[1]:
                local_groups[i]        = global_group.Range_incl([[r[0], r[1], 1]])
                local_communicators[i] = comm.Create_group(local_groups[i], i)
                owned_groups.append(i)

        try:
            carts = [CartDecomposition(n, p, P, reorder, comm=sub_comm, shifts=s, num_threads=num_threads) if sub_comm else None\
                    for n,p,P,sub_comm,s in zip(npts, pads, periods, local_communicators, shifts)]
        except:
            comm.Abort(1)

        self._local_groups        = local_groups
        self._local_communicators = local_communicators
        self._owned_groups        = owned_groups
        self._carts               = carts

    @property
    def npts( self ):
        return self._npts

    @property
    def pads( self ):
        return self._pads

    @property
    def periods( self ):
        return self._periods

    @property
    def shifts( self ):
        return self._shifts

    @property
    def num_threads( self ):
        return self._num_threads

    @property
    def comm( self ):
        return self._comm

    @property
    def ncarts( self ):
        return self._ncarts

    @property
    def size( self ):
        return self._size

    @property
    def rank( self ):
        return self._rank

    @property
    def sizes( self ):
        return self._sizes

    @property
    def rank_ranges( self ):
        return self._rank_ranges

    @property
    def local_groups( self ):
        return self._local_groups

    @property
    def local_communicators( self ):
        return self._local_communicators

    @property
    def owned_groups( self ):
        return self._owned_groups

    @property
    def carts( self ):
        return self._carts

#===============================================================================
class InterfacesCartDecomposition:
    def __init__(self, carts, interfaces):

        assert isinstance(carts, MultiCartDecomposition)

        npts                = carts.npts
        pads                = carts.pads
        shifts              = carts.shifts
        periods             = carts.periods
        num_threads         = carts.num_threads
        comm                = carts.comm
        global_group        = comm.group
        local_groups        = carts.local_groups
        rank_ranges         = carts.rank_ranges
        local_communicators = carts.local_communicators
        owned_groups        = carts.owned_groups

        interfaces_groups     = {}
        interfaces_comm       = {}
        interfaces_root_ranks = {}
        interfaces_carts      = {}

        for i,j in interfaces:
            if i in owned_groups or j in owned_groups:
                if not local_groups[i]:
                    local_groups[i] = global_group.Range_incl([[rank_ranges[i][0], rank_ranges[i][1], 1]])
                if not local_groups[j]:
                    local_groups[j] = global_group.Range_incl([[rank_ranges[j][0], rank_ranges[j][1], 1]])

                interfaces_groups[i,j] = local_groups[i].Union(local_groups[i], local_groups[j])
                interfaces_comm  [i,j] = comm.Create_group(interfaces_groups[i,j])
                root_rank_i            = interfaces_groups[i,j].Translate_ranks(local_groups[i], [0], interfaces_groups[i,j])[0]
                root_rank_j            = interfaces_groups[i,j].Translate_ranks(local_groups[j], [0], interfaces_groups[i,j])[0]
                interfaces_root_ranks[i,j] = [root_rank_i, root_rank_j]

        tag   = lambda i,j,disp: (2+disp)*(i+j)
        dtype = find_mpi_type('int64')
        for i,j in interfaces:
            if (i,j) in interfaces_comm:
                ranks_in_topo_i = carts.carts[i].ranks_in_topo if i in owned_groups else np.full(local_groups[i].size, -1)
                ranks_in_topo_j = carts.carts[j].ranks_in_topo if j in owned_groups else np.full(local_groups[j].size, -1)
                req = []
                if interfaces_comm[i,j].rank == interfaces_root_ranks[i,j][0]:
                    req.append(interfaces_comm[i,j].Isend((ranks_in_topo_i, ranks_in_topo_i.size, dtype), interfaces_root_ranks[i,j][1], tag=tag(i,j,1)))
                    req.append(interfaces_comm[i,j].Irecv((ranks_in_topo_j, ranks_in_topo_j.size, dtype), interfaces_root_ranks[i,j][1], tag=tag(i,j,-1)))

                if interfaces_comm[i,j].rank == interfaces_root_ranks[i,j][1]:
                    req.append(interfaces_comm[i,j].Isend((ranks_in_topo_j, ranks_in_topo_j.size, dtype), interfaces_root_ranks[i,j][0], tag=tag(i,j,-1)))
                    req.append(interfaces_comm[i,j].Irecv((ranks_in_topo_i, ranks_in_topo_i.size, dtype), interfaces_root_ranks[i,j][0], tag=tag(i,j,1)))                  

                axes = interfaces[i,j][0]
                exts = interfaces[i,j][1]
                interfaces_carts[i,j] = InterfaceCartDecomposition(npts=[npts[i], npts[j]],
                                                                   pads=[pads[i], pads[j]],
                                                                   periods=[periods[i], periods[j]],
                                                                   comm=interfaces_comm[i,j],
                                                                   shifts=[shifts[i], shifts[j]],
                                                                   axes=axes, exts=exts, 
                                                                   ranks_in_topo=[ranks_in_topo_i, ranks_in_topo_j],
                                                                   local_groups=[local_groups[i], local_groups[j]],
                                                                   local_communicators=[local_communicators[i], local_communicators[j]],
                                                                   root_ranks=interfaces_root_ranks[i,j],
                                                                   requests=req,
                                                                   num_threads=num_threads)

        self._interfaces_groups = interfaces_groups
        self._interfaces_comm   = interfaces_comm
        self._carts             = interfaces_carts

    @property
    def carts( self ):
        return self._carts

    @property
    def interfaces_comm( self ):
        return self._interfaces_comm

    @property
    def interfaces_groups( self ):
        return self._interfaces_groups

#===============================================================================
class InterfaceCartDecomposition:

    def __init__(self, npts, pads, periods, comm, shifts, axes, exts, ranks_in_topo, local_groups, local_communicators, root_ranks, requests, num_threads):

        npts_minus, npts_plus       = npts
        pads_minus, pads_plus       = pads
        periods_minus, periods_plus = periods
        shifts_minus, shifts_plus   = shifts
        axis_minus, axis_plus       = axes
        ext_minus, ext_plus         = exts
        size_minus, size_plus       = len(ranks_in_topo[0]), len(ranks_in_topo[1])

        assert axis_minus == axis_plus

        root_rank_minus, root_rank_plus         = root_ranks
        local_comm_minus, local_comm_plus       = local_communicators
        ranks_in_topo_minus, ranks_in_topo_plus = ranks_in_topo

        self._ndims         = len( npts_minus )
        self._npts_minus    = npts_minus
        self._npts_plus     = npts_plus
        self._pads_minus    = pads_minus
        self._pads_plus     = pads_plus
        self._shifts_minus  = shifts_minus
        self._shifts_plus   = shifts_plus
        self._axis          = axis_minus
        self._ext_minus     = ext_minus
        self._ext_plus      = ext_plus
        self._comm          = comm
        self._local_comm_minus  = local_communicators[0]
        self._local_comm_plus  = local_communicators[1]
        self._local_rank_minus  = None
        self._local_rank_plus  = None

        if self._local_comm_minus:
            self._local_rank_minus = self._local_comm_minus.rank

        if self._local_comm_plus:
            self._local_rank_plus = self._local_comm_plus.rank

        reduced_npts_minus = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts_minus, shifts_minus, pads_minus, periods_minus)]
        reduced_npts_plus = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts_plus, shifts_plus, pads_plus, periods_plus)]

        nprocs_minus, block_shape_minus = compute_dims( size_minus, reduced_npts_minus, pads_minus )
        nprocs_plus, block_shape_plus   = compute_dims( size_plus, reduced_npts_plus, pads_plus )

        dtype = find_mpi_type('int64')
        MPI.Request.Waitall(requests)

        if local_comm_minus:
            local_comm_minus.Bcast((ranks_in_topo_plus,ranks_in_topo_plus.size, dtype), root=0)

        if local_comm_plus:
            local_comm_plus.Bcast((ranks_in_topo_minus,ranks_in_topo_minus.size, dtype), root=0)

        self._coords_from_rank_minus = np.array([np.unravel_index(rank, nprocs_minus) for rank in range(size_minus)])
        self._coords_from_rank_plus = np.array([np.unravel_index(rank, nprocs_plus) for rank in range(size_plus)])

        rank_from_coords_minus = np.zeros(nprocs_minus, dtype=int)
        rank_from_coords_plus = np.zeros(nprocs_plus, dtype=int)

        for r in range(size_minus):
            rank_from_coords_minus[tuple(self._coords_from_rank_minus[r])] = r

        for r in range(size_plus):
            rank_from_coords_plus[tuple(self._coords_from_rank_plus[r])] = r

        index_minus      = [slice(None, None)]*len(npts_minus)
        index_plus       = [slice(None, None)]*len(npts_minus)
        index_minus[axis_minus] = 0 if ext_minus == -1 else -1
        index_plus[axis_plus]  = 0 if ext_plus == -1 else -1

        self._boundary_ranks_minus = rank_from_coords_minus[tuple(index_minus)].ravel()
        self._boundary_ranks_plus  = rank_from_coords_plus[tuple(index_plus)].ravel()

        boundary_group_minus = local_groups[0].Incl(self._boundary_ranks_minus)
        boundary_group_plus  = local_groups[1].Incl(self._boundary_ranks_plus)

        comm_minus = comm.Create_group(boundary_group_minus)
        comm_plus  = comm.Create_group(boundary_group_plus)

        root_minus = comm.group.Translate_ranks(boundary_group_minus, [0], comm.group)[0]
        root_plus  = comm.group.Translate_ranks(boundary_group_plus, [0], comm.group)[0]

        procs_index_minus = boundary_group_minus.Translate_ranks(local_groups[0], self._boundary_ranks_minus, boundary_group_minus)
        procs_index_plus  = boundary_group_plus.Translate_ranks(local_groups[1],  self._boundary_ranks_plus,  boundary_group_plus)

        # Reorder procs ranks from 0 to local_group.size-1
        self._boundary_ranks_minus = self._boundary_ranks_minus[procs_index_minus]
        self._boundary_ranks_plus  = self._boundary_ranks_plus[procs_index_plus]
        
        if root_minus != root_plus:
            if not comm_minus == MPI.COMM_NULL:
                intercomm = comm_minus.Create_intercomm(0, comm, root_plus)
                

            elif not comm_plus == MPI.COMM_NULL:
                intercomm = comm_plus.Create_intercomm(0, comm, root_minus)
            else:
                intercomm = None
        else:
            intercomm = None

        self._intercomm = intercomm

        # Store arrays with all the reduced starts and reduced ends along each direction
        self._reduced_global_starts_minus = [None]*self._ndims
        self._reduced_global_ends_minus   = [None]*self._ndims
        self._reduced_global_starts_plus  = [None]*self._ndims
        self._reduced_global_ends_plus    = [None]*self._ndims
        for axis in range( self._ndims ):
            ni = reduced_npts_minus[axis]
            di = nprocs_minus[axis]
            pi = pads_minus[axis]
            mi = shifts_minus[axis]
            nj = reduced_npts_plus[axis]
            dj = nprocs_plus[axis]
            pj = pads_plus[axis]
            mj = shifts_plus[axis]

            self._reduced_global_starts_minus[axis] = np.array( [( ci   *ni)//di   for ci in range( di )] )
            self._reduced_global_ends_minus  [axis] = np.array( [((ci+1)*ni)//di-1 for ci in range( di )] )
            self._reduced_global_starts_plus[axis] = np.array( [( cj   *nj)//dj   for cj in range( dj )] )
            self._reduced_global_ends_plus  [axis] = np.array( [((cj+1)*nj)//dj-1 for cj in range( dj )] )
            if mi>1:self._reduced_global_ends_minus [axis][-1] += pi+1
            if mj>1:self._reduced_global_ends_plus [axis][-1] += pj+1

        # Store arrays with all the starts and ends along each direction
        self._global_starts_minus = [None]*self._ndims
        self._global_ends_minus   = [None]*self._ndims
        self._global_starts_plus = [None]*self._ndims
        self._global_ends_plus   = [None]*self._ndims

        for axis in range( self._ndims ):
            ni = npts_minus[axis]
            di = nprocs_minus[axis]
            pi = pads_minus[axis]
            mi = shifts_minus[axis]
            r_starts_minus = self._reduced_global_starts_minus[axis]
            r_ends_minus   = self._reduced_global_ends_minus  [axis]
            nj = npts_plus[axis]
            dj = nprocs_plus[axis]
            pj = pads_plus[axis]
            mj = shifts_plus[axis]
            r_starts_plus = self._reduced_global_starts_plus[axis]
            r_ends_plus   = self._reduced_global_ends_plus  [axis]

            global_starts_minus = [0]
            for ci in range(1,di):
                global_starts_minus.append(global_starts_minus[ci-1] + (r_ends_minus[ci-1]-r_starts_minus[ci-1]+1)*mi)

            global_starts_plus = [0]
            for cj in range(1,dj):
                global_starts_plus.append(global_starts_plus[cj-1] + (r_ends_plus[cj-1]-r_starts_plus[cj-1]+1)*mj)

            global_ends_minus   = [global_starts_minus[ci+1]-1 for ci in range( di-1 )] + [ni-1]
            global_ends_plus   = [global_starts_plus[cj+1]-1 for cj in range( dj-1 )] + [nj-1]

            self._global_starts_minus[axis] = np.array( global_starts_minus )
            self._global_ends_minus  [axis] = np.array( global_ends_minus )
            self._global_starts_plus[axis] = np.array( global_starts_plus )
            self._global_ends_plus  [axis] = np.array( global_ends_plus )

        self._communication_infos = {}
        self._communication_infos[self._axis] = self._compute_communication_infos(self._axis)


    @property
    def ndims( self ):
        return self._ndims

    @property
    def npts_minus( self ):
        return self._npts_minus

    @property
    def npts_plus( self ):
        return self._npts_plus

    @property
    def pads_minus( self ):
        return self._pads_minus

    @property
    def pads_plus( self ):
        return self._pads_plus

    @property
    def shifts_minus( self ):
        return self._shifts_minus

    @property
    def shifts_plus( self ):
        return self._shifts_plus

    @property
    def ext_minus( self ):
        return self._ext_minus

    @property
    def ext_plus( self ):
        return self._ext_plus

    @property
    def local_comm_minus( self ):
        return self._local_comm_minus

    @property
    def local_comm_plus( self ):
        return self._local_comm_plus

    @property
    def local_rank_minus( self ):
        return self._local_rank_minus

    @property
    def local_rank_plus( self ):
        return self._local_rank_plus

    @property
    def coords_from_rank_minus( self ):
        return self._coords_from_rank_minus

    @property
    def coords_from_rank_plus( self ):
        return self._coords_from_rank_plus

    @property
    def boundary_ranks_minus( self ):
        return self._boundary_ranks_minus

    @property
    def boundary_ranks_plus( self ):
        return self._boundary_ranks_plus

    @property
    def reduced_global_starts_minus( self ):
        return self._reduced_global_starts_minus

    @property
    def reduced_global_starts_plus( self ):
        return self._reduced_global_starts_plus

    @property
    def reduced_global_ends_minus( self ):
        return self._reduced_global_ends_minus

    @property
    def reduced_global_ends_plus( self ):
        return self._reduced_global_ends_plus

    @property
    def global_starts_minus( self ):
        return self._global_starts_minus

    @property
    def global_starts_plus( self ):
        return self._global_starts_plus

    @property
    def global_ends_minus( self ):
        return self._global_ends_minus

    @property
    def global_ends_plus( self ):
        return self._global_ends_plus

    @property
    def axis( self ):
        return self._axis

    @property
    def comm( self ):
        return self._comm

    @property
    def intercomm( self ):
        return self._intercomm

    def get_communication_infos( self, axis ):
        return self._communication_infos[ axis ]

    #---------------------------------------------------------------------------
    def _compute_communication_infos( self, axis ):

        if self._intercomm == MPI.COMM_NULL:
            return 

        # Mesh info
        npts_minus   = self._npts_minus
        npts_plus    = self._npts_plus
        pads_minus   = self._pads_minus
        pads_plus    = self._pads_plus
        shifts_minus = self._shifts_minus
        shifts_plus  = self._shifts_plus
        ext_minus    = self._ext_minus
        ext_plus     = self._ext_plus
        indices  = []

        if self._local_rank_minus is not None:
            rank_minus = self._local_rank_minus
            coords = self._coords_from_rank_minus[rank_minus]
            starts = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]
            send_shape   = [e-s+1+2*m*p for s,e,m,p in zip(starts, ends, shifts_minus, pads_minus)]
            send_starts  = [m*p for m,p in zip(shifts_minus, pads_minus)]
            m,p,s,e      = shifts_minus[axis], pads_minus[axis], starts[axis], ends[axis]
            send_starts[axis] = m*p if ext_minus == -1 else m*p+e-s+1-p-1
            starts[axis] = starts[axis] if ext_minus == -1 else ends[axis]-pads_minus[axis]
            ends[axis]   = starts[axis]+pads_minus[axis] if ext_minus == -1 else ends[axis]
            send_buf_shape    = [e-s+1 for s,e,p,m in zip(starts, ends, pads_minus, shifts_minus)]
            # ...
            coords = self._coords_from_rank_plus[self._boundary_ranks_plus[0]]
            starts = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]

            recv_shape       = [n+2*m*p for n,m,p in zip(npts_plus, shifts_plus, pads_plus)]
            recv_shape[axis] = pads_plus[axis]+1 + 2*shifts_plus[axis]*pads_plus[axis]

            displacements   = [0]*(len(self._boundary_ranks_plus)+1)
            recv_counts     = [None]*len(self._boundary_ranks_plus)
            for k,b in enumerate(self._boundary_ranks_plus):
                coords = self._coords_from_rank_plus[b]
                starts = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]
                starts[axis] = starts[axis] if ext_plus == -1 else ends[axis]-pads_plus[axis]
                ends[axis]   = starts[axis]+pads_plus[axis] if ext_plus == -1 else ends[axis]
                shape_k = [e-s+1 for s,e in zip(starts, ends)]
                recv_counts[k] = np.product(shape_k)
                ranges       = [(s+p*m, p*m+e+1) for s,e,p,m in zip(starts, ends, pads_plus, shifts_plus)]
                ranges[axis] = (shifts_plus[axis]*pads_plus[axis], shifts_plus[axis]*pads_plus[axis]+shape_k[axis])
                indices     += [np.ravel_multi_index( ii, dims=recv_shape, order='C' ) for ii in itertools.product(*[range(*a) for a in ranges])] 

        elif self._local_rank_plus is not None:
            rank_plus = self._local_rank_plus
            coords = self._coords_from_rank_plus[rank_plus]
            starts = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]
            send_shape   = [e-s+1+2*m*p for s,e,m,p in zip(starts, ends, shifts_plus, pads_plus)]
            send_starts  = [m*p for m,p in zip(shifts_plus, pads_plus)]
            m,p,s,e = shifts_plus[axis], pads_plus[axis], starts[axis], ends[axis]
            send_starts[axis] = m*p if ext_plus == -1 else m*p+e-s+1-p-1
            starts[axis] = starts[axis] if ext_plus == -1 else ends[axis]-pads_plus[axis]
            ends[axis]   = starts[axis]+pads_plus[axis] if ext_plus == -1 else ends[axis]
            send_buf_shape  = [e-s+1 for s,e,p,m in zip(starts, ends, pads_plus, shifts_plus)]

            # ...
            coords = self._coords_from_rank_minus[self._boundary_ranks_minus[0]]
            starts = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]

            recv_shape           = [n+2*m*p for n,m,p in zip(npts_minus, shifts_minus, pads_minus)]
            recv_shape[axis]     = pads_minus[axis]+1 + 2*shifts_minus[axis]*pads_minus[axis]

            displacements   = [0]*(len(self._boundary_ranks_minus)+1)
            recv_counts     = [None]*len(self._boundary_ranks_minus)
            for k,b in enumerate(self._boundary_ranks_minus):
                coords = self._coords_from_rank_minus[b]
                starts = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]
                starts[axis] = starts[axis] if ext_minus == -1 else ends[axis]-pads_minus[axis]
                ends[axis]   = starts[axis]+pads_minus[axis] if ext_minus == -1 else ends[axis]
                shape_k = [e-s+1 for s,e in zip(starts, ends)]
                recv_counts[k] = np.product(shape_k)
                ranges       = [(s+p*m, p*m+e+1) for s,e,p,m in zip(starts, ends, pads_minus, shifts_minus)]
                ranges[axis] = (shifts_minus[axis]*pads_minus[axis], shifts_minus[axis]*pads_minus[axis]+shape_k[axis])
                indices     += [np.ravel_multi_index( ii, dims=recv_shape, order='C' ) for ii in itertools.product(*[range(*a) for a in ranges])] 

        displacements[1:] = np.cumsum(recv_counts)
        # Store all information into dictionary
        info = {'send_buf_shape' : tuple( send_buf_shape ),
                'send_starts'    : tuple( send_starts ),
                'send_shape'     : tuple( send_shape  ),
                'recv_shape'     : tuple( recv_shape ),
                'displacements'  : tuple( displacements ),
                'recv_counts'    : tuple( recv_counts),
                'indices'        : indices}

        return info

#===============================================================================
class CartDecomposition():
    """
    Cartesian decomposition of a tensor-product grid of spline coefficients.
    This is built on top of an MPI communicator with multi-dimensional
    Cartesian topology.

    Parameters
    ----------
    npts : list or tuple of int
        Number of coefficients in the global grid along each dimension.

    pads : list or tuple of int
        Padding along each grid dimension.
        In 1D, this is the number of extra coefficients added at each boundary
        of the local domain to permit non-local operations with compact support;
        this concept extends to multiple dimensions through a tensor product.

    periods : list or tuple of bool
        Periodicity (True|False) along each grid dimension.

    reorder : bool
        Whether individual ranks can be changed in the new Cartesian communicator.

    comm : mpi4py.MPI.Comm
        MPI communicator that will be used to spawn a new Cartesian communicator
        (optional: default is MPI_COMM_WORLD).

    shifts: list or tuple of int
        Shifts along each grid dimension.
        It takes values bigger or equal to one, it represents the multiplicity of each knot.

    nprocs: list or tuple of int
       MPI decomposition along each dimension.

    reverse_axis: int
       Reverse the ownership of the processes along the specified axis.

    """
    def __init__( self, npts, pads, periods, reorder, comm=None, shifts=None, nprocs=None, reverse_axis=None, num_threads=None ):

        # Check input arguments
        # TODO: check that arguments are identical across all processes
        assert len( npts ) == len( pads ) == len( periods )
        assert all( n >=1 for n in npts )
        assert all( p >=0 for p in pads )
        assert all( isinstance( period, bool ) for period in periods )
        assert isinstance( reorder, bool )
        assert isinstance( comm, MPI.Comm )

        shifts      = tuple(shifts) if shifts else (1,)*len(npts)
        num_threads = num_threads if num_threads else 1

        # Store input arguments
        self._npts         = tuple( npts    )
        self._pads         = tuple( pads    )
        self._periods      = tuple( periods )
        self._shifts       = shifts
        self._num_threads  = num_threads
        self._reorder      = reorder
        self._comm         = comm

        # ...
        self._ndims = len( npts )
        # ...
        # ...
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        # ...

        # ...
        # Know the number of processes along each direction
#        self._dims = MPI.Compute_dims( self._size, self._ndims )

        reduced_npts = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts, shifts, pads, periods)]

        if nprocs is None:
            nprocs, block_shape = compute_dims( self._size, reduced_npts, pads )
        else:
            assert len(nprocs) == len(npts)

        assert np.product(nprocs) == self._size

        self._dims = nprocs
        self._reverse_axis = reverse_axis
        # ...

        # ...
        # Create a 2D MPI cart
        self._comm_cart = comm.Create_cart(
            dims    = self._dims,
            periods = self._periods,
            reorder = self._reorder
        )

        # Know my coordinates in the topology
        self._rank_in_topo  = self._comm_cart.Get_rank()
        self._coords        = self._comm_cart.Get_coords( rank=self._rank_in_topo )
        self._ranks_in_topo = np.array(comm.group.Translate_ranks(self._comm_cart.group, list(range(self._comm_cart.size)), comm.group))

        if reverse_axis is not None:
            self._coords[reverse_axis] = self._dims[reverse_axis] - self._coords[reverse_axis] - 1

        # Store arrays with all the reduced starts and reduced ends along each direction
        self._reduced_global_starts = [None]*self._ndims
        self._reduced_global_ends   = [None]*self._ndims
        for axis in range( self._ndims ):
            n = reduced_npts[axis]
            d = nprocs[axis]
            p = pads[axis]
            m = shifts[axis]
            self._reduced_global_starts[axis] = np.array( [( c   *n)//d   for c in range( d )] )
            self._reduced_global_ends  [axis] = np.array( [((c+1)*n)//d-1 for c in range( d )] )
            if m>1:self._reduced_global_ends  [axis][-1] += p+1

        # Store arrays with all the starts and ends along each direction
        self._global_starts = [None]*self._ndims
        self._global_ends   = [None]*self._ndims

        for axis in range( self._ndims ):
            n = npts[axis]
            d = nprocs[axis]
            p = pads[axis]
            m = shifts[axis]
            r_starts = self._reduced_global_starts[axis]
            r_ends   = self._reduced_global_ends  [axis]

            global_starts = [0]
            for c in range(1,d):
                global_starts.append(global_starts[c-1] + (r_ends[c-1]-r_starts[c-1]+1)*m)

            global_ends = [global_starts[c+1]-1 for c in range( d-1 )] + [n-1]

            self._global_starts[axis] = np.array( global_starts )
            self._global_ends  [axis] = np.array( global_ends )

        # Start/end values of global indices (without ghost regions)
        self._starts = tuple( self.global_starts[axis][c] for axis,c in zip(range(self._ndims), self._coords) )
        self._ends   = tuple( self.global_ends  [axis][c] for axis,c in zip(range(self._ndims), self._coords) )

        # List of 1D global indices (without ghost regions)
        self._grids = tuple( range(s,e+1) for s,e in zip( self._starts, self._ends ) )

        # Compute shape of local arrays in topology (with ghost regions)
        self._shape = tuple( e-s+1+2*m*p for s,e,p,m in zip( self._starts, self._ends, self._pads, shifts ) )

        # Extended grids with ghost regions
        self._extended_grids = tuple( range(s-m*p,e+m*p+1) for s,e,p,m in zip( self._starts, self._ends, self._pads, shifts ) )

        # Create (N-1)-dimensional communicators within the Cartesian topology
        self._subcomm = [None]*self._ndims
        for i in range(self._ndims):
            remain_dims     = [i==j for j in range( self._ndims )]
            self._subcomm[i] = self._comm_cart.Sub( remain_dims )

        # Compute/store information for communicating with neighbors
        self._shift_info = {}
        for axis in range( self._ndims ):
            for disp in [-1,1]:
                self._shift_info[ axis, disp ] = \
                        self._compute_shift_info( axis, disp )

        self._petsccart     = None
        self._parent_starts = tuple([None]*self._ndims)
        self._parent_ends   = tuple([None]*self._ndims)
    #---------------------------------------------------------------------------
    # Global properties (same for each process)
    #---------------------------------------------------------------------------
    @property
    def ndim( self ):
        return self._ndims

    @property
    def npts( self ):
        return self._npts

    @property
    def pads( self ):
        return self._pads

    @property
    def periods( self ):
        return self._periods

    @property
    def shifts( self ):
        return self._shifts

    @property
    def num_threads( self ):
        return self._num_threads

    @property
    def reorder( self ):
        return self._reorder

    @property
    def comm( self ):
        return self._comm

    @property
    def comm_cart( self ):
        return self._comm_cart

    @property
    def nprocs( self ):
        return self._dims

    @property
    def reverse_axis(self):
        return self._reverse_axis

    @property
    def global_starts( self ):
        return self._global_starts

    @property
    def global_ends( self ):
        return self._global_ends

    @property
    def reduced_global_starts( self ):
        return self._reduced_global_starts

    @property
    def reduced_global_ends( self ):
        return self._reduced_global_ends

    @property
    def ranks_in_topo( self ):
        return self._ranks_in_topo

    #---------------------------------------------------------------------------
    # Local properties
    #---------------------------------------------------------------------------
    @property
    def starts( self ):
        return self._starts

    @property
    def ends( self ):
        return self._ends

    @property
    def parent_starts( self ):
        return self._starts

    @property
    def parent_ends( self ):
        return self._parent_ends

    @property
    def coords( self ):
        return self._coords

    @property
    def shape( self ):
        return self._shape

    @property
    def subcomm( self ):
        return self._subcomm

# NOTE [YG, 09.03.2021]: the equality comparison "==" is removed because we
# prefer using the identity comparison "is" as far as possible.
#    def __eq__( self, a):
#        a = (a.npts, a.pads, a.periods, a.comm)
#        b = (self.npts, self.pads, self.periods, self.comm)
#        return a == b

    #---------------------------------------------------------------------------
    def topetsc( self ):
        """ Convert the cart to a petsc cart.
        """
        if self._petsccart is None:
            from psydac.ddm.petsc import PetscCart
            self._petsccart = PetscCart(self)
        return self._petsccart

    #---------------------------------------------------------------------------
    def coords_exist( self, coords ):

        return all( P or (0 <= c < d) for P,c,d in zip( self._periods, coords, self._dims ) )

    #---------------------------------------------------------------------------
    def get_shift_info( self, direction, disp ):

        return self._shift_info[ direction, disp ]

    #---------------------------------------------------------------------------
    def get_shared_memory_subdivision( self, shape ):

        assert len(shape) == self._ndims

        try:
            nthreads , block_shape = compute_dims( self._num_threads, shape , [2*p for p in self._pads])
        except ValueError:
            print("Cannot compute dimensions with given input values!")
            self.comm.Abort(1)

        # compute the coords for all threads
        coords_from_rank = np.array([np.unravel_index(rank, nthreads) for rank in range(self._num_threads)])
        rank_from_coords = np.zeros([n+1 for n in nthreads], dtype=int)
        for r in range(self._num_threads):
            c = coords_from_rank[r]
            rank_from_coords[tuple(c)] = r

        # rank_from_coords is not used in the current version of the assembly code
        # it's used in the commented second version, where we don't use a global barrier, but needs more checks to work

        for i in range(self._ndims):
            ind = [slice(None,None)]*self._ndims
            ind[i] = nthreads[i]
            rank_from_coords[tuple(ind)] = self._num_threads

        # Store arrays with all the starts and ends along each direction for every thread
        thread_global_starts = [None]*self._ndims
        thread_global_ends   = [None]*self._ndims
        for axis in range( self._ndims ):
            n = shape[axis]
            d = nthreads[axis]
            thread_global_starts[axis] = np.array( [( c   *n)//d   for c in range( d )] )
            thread_global_ends  [axis] = np.array( [((c+1)*n)//d-1 for c in range( d )] )

        return coords_from_rank, rank_from_coords, thread_global_starts, thread_global_ends, self._num_threads

    #---------------------------------------------------------------------------
    def _compute_shift_info( self, direction, disp ):

        assert( 0 <= direction < self._ndims )
        assert( isinstance( disp, int ) )

        reorder = self.reverse_axis == direction
        # Process ranks for data shifting with MPI_SENDRECV
        (rank_source, rank_dest) = self.comm_cart.Shift( direction, disp )

        if reorder:
            (rank_source, rank_dest) = (rank_dest, rank_source)

        # Mesh info info along given direction
        s = self._starts[direction]
        e = self._ends  [direction]
        p = self._pads  [direction]
        m = self._shifts[direction]

        # Shape of send/recv subarrays
        buf_shape = np.array( self._shape )
        buf_shape[direction] = m*p

        # Start location of send/recv subarrays
        send_starts = np.zeros( self._ndims, dtype=int )
        recv_starts = np.zeros( self._ndims, dtype=int )
        if disp > 0:
            recv_starts[direction] = 0
            send_starts[direction] = e-s+1
        elif disp < 0:
            recv_starts[direction] = e-s+1+m*p
            send_starts[direction] = m*p

        # Store all information into dictionary
        info = {'rank_dest'  : rank_dest,
                'rank_source': rank_source,
                'buf_shape'  : tuple(  buf_shape  ),
                'send_starts': tuple( send_starts ),
                'recv_starts': tuple( recv_starts )}
        return info
        
    def reduce_elements( self, axes, n_elements):
        """ Compute the cart of the reduced space.

        Parameters
        ----------
        axes: tuple_like (int)
            The directions to be Reduced.

        n_elements: tuple_like (int)
            Number of elements to substract from the space.

        Returns
        -------
        v: CartDecomposition
            The reduced cart.
        """

        if isinstance(axes, int):
            axes = [axes]

        cart = CartDecomposition(self._npts, self._pads, self._periods, self._reorder, shifts=self.shifts, reverse_axis=self.reverse_axis)

        cart._dims      = self._dims
        cart._comm_cart = self._comm_cart
        cart._coords    = self._coords

        coords          = cart.coords
        nprocs          = cart.nprocs

        cart._shifts = [max(1,m-1) for m in self.shifts]

        for axis in axes:assert(axis<cart._ndims)

        # set pads and npts
        cart._npts = tuple(n - ne for n,ne in zip(cart.npts, n_elements))

        # Store arrays with all the starts and ends along each direction
        cart._global_starts = [None]*self._ndims
        cart._global_ends   = [None]*self._ndims
        for axis in range( self._ndims ):
            n = cart._npts[axis]
            d = nprocs[axis]
            m = cart._shifts[axis]
            r_starts = cart._reduced_global_starts[axis]
            r_ends   = cart._reduced_global_ends  [axis]

            global_starts = [0]
            for c in range(1,d):
                global_starts.append(global_starts[c-1] + (r_ends[c-1]-r_starts[c-1]+1)*m)

            global_ends = [global_starts[c+1]-1 for c in range( d-1 )] + [n-1]

            cart._global_starts[axis] = np.array( global_starts )
            cart._global_ends  [axis] = np.array( global_ends )

        # Start/end values of global indices (without ghost regions)
        cart._starts = tuple( cart.global_starts[axis][c] for axis,c in zip(range(self._ndims), self._coords) )
        cart._ends   = tuple( cart.global_ends  [axis][c] for axis,c in zip(range(self._ndims), self._coords) )

        # List of 1D global indices (without ghost regions)
        cart._grids = tuple( range(s,e+1) for s,e in zip( cart._starts, cart._ends ) )

        # N-dimensional global indices (without ghost regions)
        cart._indices = product( *cart._grids )

        # Compute shape of local arrays in topology (with ghost regions)
        cart._shape = tuple( e-s+1+2*m*p for s,e,p,m in zip( cart._starts, cart._ends, cart._pads, cart._shifts ) )

        # Extended grids with ghost regions
        cart._extended_grids = tuple( range(s-m*p,e+m*p+1) for s,e,p,m in zip( cart._starts, cart._ends, cart._pads, cart._shifts ) )

        # N-dimensional global indices with ghost regions
        cart._extended_indices = product( *cart._extended_grids )

        # Create (N-1)-dimensional communicators within the cartsian topology
        cart._subcomm = [None]*cart._ndims
        for i in range(cart._ndims):
            remain_dims      = [i==j for j in range( cart._ndims )]
            cart._subcomm[i] = cart._comm_cart.Sub( remain_dims )

        # Compute/store information for communicating with neighbors
        cart._shift_info = {}
        for axis in range( cart._ndims ):
            for disp in [-1,1]:
                cart._shift_info[ axis, disp ] = \
                        cart._compute_shift_info( axis, disp )

        # Store arrays with all the reduced starts and reduced ends along each direction
        cart._reduced_global_starts = [None]*self._ndims
        cart._reduced_global_ends   = [None]*self._ndims
        for axis in range( self._ndims ):
            cart._reduced_global_starts[axis] = self._reduced_global_starts[axis].copy()
            cart._reduced_global_ends  [axis] = self._reduced_global_ends  [axis].copy()

            # adjust only the end of the last interval
            if not cart.periods[axis]:
                n = cart._npts[axis]
                cart._reduced_global_ends[axis][-1] = n-1

        cart._parent_starts = self.starts
        cart._parent_ends   = self.ends
        return cart

    def reduce_grid(self, global_starts, global_ends):
        """ 
        Returns a new CartDecomposition object with a coarser grid from the original one
        we do that by giving a new global_starts  and global_ends of the coefficients
        in each dimension.
            
        Parameters
        ----------
        global_starts : list/tuple
            the list of the new global_starts  in each dimesion.

        global_ends : list/tuple
            the list of the new global_ends in each dimesion.
 
        """
        # Make a copy
        cart = CartDecomposition(self.npts, self.pads, self.periods, self.reorder, comm=self.comm)

        cart._npts = tuple(end[-1] + 1 for end in global_ends)

        cart._dims = self._dims

        # Create a 2D MPI cart
        cart._comm_cart = self._comm_cart

        # Know my coordinates in the topology
        cart._rank_in_topo = self._rank_in_topo
        cart._coords       = self._coords

        # Start/end values of global indices (without ghost regions)
        cart._starts = tuple( starts[i] for i,starts in zip( self._coords, global_starts) )
        cart._ends   = tuple( ends[i]   for i,ends   in zip( self._coords, global_ends  ) )

        # List of 1D global indices (without ghost regions)
        cart._grids = tuple( range(s,e+1) for s,e in zip( cart._starts, cart._ends ) )

        # N-dimensional global indices (without ghost regions)
        cart._indices = product( *cart._grids )

        # Compute shape of local arrays in topology (with ghost regions)
        cart._shape = tuple( e-s+1+2*p for s,e,p in zip( cart._starts, cart._ends, cart._pads ) )

        # Extended grids with ghost regions
        cart._extended_grids = tuple( range(s-p,e+p+1) for s,e,p in zip( cart._starts, cart._ends, cart._pads ) )

        # N-dimensional global indices with ghost regions
        cart._extended_indices = product( *cart._extended_grids )

        # Compute/store information for communicating with neighbors
        cart._shift_info = {}
        for dimension in range( cart._ndims ):
            for disp in [-1,1]:
                cart._shift_info[ dimension, disp ] = \
                        cart._compute_shift_info( dimension, disp )

        # Store arrays with all the starts and ends along each direction
        cart._global_starts = global_starts
        cart._global_ends   = global_ends

        return cart

#===============================================================================
class CartDataExchanger:
    """
    Type that takes care of updating the ghost regions (padding) of a
    multi-dimensional array distributed according to the given Cartesian
    decomposition of a tensor-product grid of coefficients.

    Each coefficient in the decomposed grid may have multiple components,
    contiguous in memory.

    Parameters
    ----------
    cart : psydac.ddm.CartDecomposition
        Object that contains all information about the Cartesian decomposition
        of a tensor-product grid of coefficients.

    dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
        Datatype of single coefficient (if scalar) or of each of its
        components (if vector).

    coeff_shape : [tuple(int) | list(int)]
        Shape of a single coefficient, if this is multi-dimensional
        (optional: by default, we assume scalar coefficients).

    """
    def __init__( self, cart, dtype, *, coeff_shape=() ):

        self._send_types, self._recv_types = self._create_buffer_types(
                cart, dtype, coeff_shape=coeff_shape )

        self._cart = cart
        self._comm = cart.comm_cart

    #---------------------------------------------------------------------------
    # Public interface
    #---------------------------------------------------------------------------
    def get_send_type( self, direction, disp ):
        return self._send_types[direction, disp]

    # ...
    def get_recv_type( self, direction, disp ):
        return self._recv_types[direction, disp]

    # ...
    def update_ghost_regions( self, array, *, direction=None ):
        """
        Update ghost regions in a numpy array with dimensions compatible with
        CartDecomposition (and coeff_shape) provided at initialization.

        Parameters
        ----------
        array : numpy.ndarray
            Multidimensional array corresponding to local subdomain in
            decomposed tensor grid, including padding.

        direction : int
            Index of dimension over which ghost regions should be updated
            (optional: by default all ghost regions are updated).

        """
        if direction is None:
            for d in range( self._cart.ndim ):
                self.update_ghost_regions( array, direction=d )
            return

        assert isinstance( array, np.ndarray )
        assert isinstance( direction, int )

        # Shortcuts
        cart = self._cart
        comm = self._comm

        # Choose non-negative invertible function tag(disp) >= 0
        # NOTES:
        #   . different values of disp must return different tags!
        #   . tag at receiver must match message tag at sender
        tag = lambda disp: 42+disp

        # Requests' handles
        requests = []

        # Start receiving data (MPI_IRECV)
        for disp in [-1,1]:
            info     = cart.get_shift_info( direction, disp )
            recv_typ = self.get_recv_type ( direction, disp )
            recv_buf = (array, 1, recv_typ)
            recv_req = comm.Irecv( recv_buf, info['rank_source'], tag(disp) )
            requests.append( recv_req )

        # Start sending data (MPI_ISEND)
        for disp in [-1,1]:
            info     = cart.get_shift_info( direction, disp )
            send_typ = self.get_send_type ( direction, disp )
            send_buf = (array, 1, send_typ)
            send_req = comm.Isend( send_buf, info['rank_dest'], tag(disp) )
            requests.append( send_req )

        # Wait for end of data exchange (MPI_WAITALL)
        MPI.Request.Waitall( requests )


    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------
    @staticmethod
    def _create_buffer_types( cart, dtype, *, coeff_shape=() ):
        """
        Create MPI subarray datatypes for updating the ghost regions (padding)
        of a multi-dimensional array distributed according to the given Cartesian
        decomposition of a tensor-product grid of coefficients.

        MPI requires a subarray datatype for accessing non-contiguous slices of
        a multi-dimensional array; this is a typical situation when updating the
        ghost regions.

        Each coefficient in the decomposed grid may have multiple components,
        contiguous in memory.

        Parameters
        ----------
        cart : psydac.ddm.CartDecomposition
            Object that contains all information about the Cartesian decomposition
            of a tensor-product grid of coefficients.

        dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
            Datatype of single coefficient (if scalar) or of each of its
            components (if vector).

        coeff_shape : [tuple(int) | list(int)]
            Shape of a single coefficient, if this is multidimensional
            (optional: by default, we assume scalar coefficients).

        Returns
        -------
        send_types : dict
            Dictionary of MPI subarray datatypes for SEND BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.

        recv_types : dict
            Dictionary of MPI subarray datatypes for RECEIVE BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.

        """
        assert isinstance( cart, CartDecomposition )

        mpi_type = find_mpi_type( dtype )

        # Possibly, each coefficient could have multiple components
        coeff_shape = list( coeff_shape )
        coeff_start = [0] * len( coeff_shape )

        data_shape = list( cart.shape ) + coeff_shape
        send_types = {}
        recv_types = {}

        for direction in range( cart.ndim ):
            for disp in [-1, 1]:
                info = cart.get_shift_info( direction, disp )

                buf_shape   = list( info[ 'buf_shape' ] ) + coeff_shape
                send_starts = list( info['send_starts'] ) + coeff_start
                recv_starts = list( info['recv_starts'] ) + coeff_start

                send_types[direction,disp] = mpi_type.Create_subarray(
                    sizes    = data_shape ,
                    subsizes =  buf_shape ,
                    starts   = send_starts,
                ).Commit()

                recv_types[direction,disp] = mpi_type.Create_subarray(
                    sizes    = data_shape ,
                    subsizes =  buf_shape ,
                    starts   = recv_starts,
                ).Commit()

        return send_types, recv_types

#===============================================================================
class InterfaceCartDataExchanger:

    def __init__(self, cart, dtype):
        args = self._create_buffer_types( cart, dtype )
    
        self._cart          = cart
        self._dtype         = dtype
        self._send_types    = args[0]
        self._recv_types    = args[1]
        self._displacements = args[2]
        self._recv_counts   = args[3]
        self._indices       = args[4]
        self._recv_buf      = np.empty((self._displacements[-1],), dtype=dtype)

    # ...
    def start_update_ghost_regions( self, array_minus, array_plus ):
        cart          = self._cart
        send_type     = self._send_types
        recv_type     = self._recv_types
        recv_buf      = self._recv_buf
        displacements = self._displacements
        recv_counts   = self._recv_counts
        indices       = self._indices
        intercomm     = cart._intercomm

        raveled_array_minus = array_minus.ravel()
        raveled_array_plus = array_plus.ravel()

        if cart._local_rank_minus is not None:
            req = intercomm.Iallgatherv([raveled_array_minus, 1, send_type],[recv_buf, recv_counts, displacements[:-1], recv_type] )

        elif cart._local_rank_plus is not None:
            req = intercomm.Iallgatherv([raveled_array_plus, 1, send_type],[recv_buf, recv_counts, displacements[:-1], recv_type] )

        return req

    def end_update_ghost_regions(self, req, array_minus, array_plus):
        cart                = self._cart
        indices             = self._indices
        recv_buf            = self._recv_buf
        raveled_array_minus = array_minus.ravel()
        raveled_array_plus  = array_plus.ravel()

        MPI.Request.wait(req)
        if cart._local_rank_minus is not None:
            raveled_array_plus[indices] = recv_buf
        elif cart._local_rank_plus is not None:
            raveled_array_minus[indices] = recv_buf


    @staticmethod
    def _create_buffer_types( cart, dtype ):

        assert isinstance( cart, InterfaceCartDecomposition )

        mpi_type = find_mpi_type( dtype )
        info     = cart.get_communication_infos( cart._axis )

        send_data_shape  = list( info['send_shape' ] )
        send_buf_shape   = list( info['send_buf_shape' ] )
        send_starts      = list( info['send_starts'] )

        send_types = mpi_type.Create_subarray(
                     sizes    = send_data_shape ,
                     subsizes =  send_buf_shape ,
                     starts   = send_starts).Commit()

        displacements    = info['displacements']
        recv_counts      = info['recv_counts']
        recv_types       = mpi_type

        return send_types, recv_types, displacements, recv_counts, info['indices']

