# pushing.pyx
# Cython wrapper for pushing.h

from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free
import numpy as np


cdef extern from "pushing.h":
    ctypedef struct PushingOpts:
        int      fine
        int      random
        uint64_t seed

    int c_pushing "pushing"(
        int        *vecs,
        int         dim,
        int         num_vecs,
        PushingOpts *opts,
        int         max_num_simps,
        uint32_t   *simps,
        int        *num_simps
    )


def pushing(pts, bint fine=True, bint random=True, seed=0,
            int max_num_simps=-1) -> tuple:
    """
    Generate a pushing triangulation of a lattice vector/point configuration.

    Parameters
    ----------
    pts : array-like of shape (n, d), int
        Input vectors. For point configurations (polytopes), pass homogenized
        coordinates: prepend or append a column of ones so each point becomes
        a d-dimensional vector. For example, the 2D square [[0,0],[1,0],[0,1],
        [1,1]] becomes [[0,0,1],[1,0,1],[0,1,1],[1,1,1]].
    fine : bool, optional
        Require a fine triangulation — one that uses every input vector
        (default: True). Implies random=True. May deadlock for general VCs;
        always succeeds for acyclic VCs (point configurations).
    random : bool, optional
        Randomize the pushing order (default: True). Required if fine=True.
    seed : int, optional
        RNG seed (default: 0). Only used when random=True.
    max_num_simps : int, optional
        Maximum number of simplices to allocate. Defaults to 3 * n.

    Returns
    -------
    simps : ndarray of shape (num_simps, d), dtype uint32
        Simplices of the triangulation. Each row contains d vector indices
        (0-indexed) into pts.
    status : int
        Status code:
             0: success
            -1: misconfigured options (fine=True requires random=True)
            -2: memory allocation problem
            -3: zero vectors input
            -4: couldn't find initial simplex
            -5: deadlock — couldn't add a new simplex (fine VC mode only)
            -6: constructed too many simplices
          -100: error splitting a cone
    """
    pts_np = np.ascontiguousarray(pts, dtype=np.int32)
    if pts_np.ndim != 2:
        raise ValueError(f"pts must be 2-D, got shape {pts_np.shape}")

    cdef int num_vecs = pts_np.shape[0]
    cdef int dim      = pts_np.shape[1]

    if max_num_simps < 0:
        max_num_simps = 3 * num_vecs

    cdef PushingOpts opts
    opts.fine   = 1 if fine   else 0
    opts.random = 1 if random else 0
    opts.seed   = <uint64_t>seed

    cdef uint32_t *simps_buf = <uint32_t *>malloc(
        max_num_simps * dim * sizeof(uint32_t))
    if simps_buf == NULL:
        raise MemoryError("Failed to allocate simps buffer")

    cdef int[:, ::1] pts_view = pts_np
    cdef int *pts_ptr = &pts_view[0, 0]

    cdef int num_simps = 0
    cdef int status = c_pushing(
        pts_ptr, dim, num_vecs,
        &opts,
        max_num_simps,
        simps_buf,
        &num_simps
    )

    out = np.empty((num_simps, dim), dtype=np.uint32)
    cdef uint32_t[:, ::1] out_view = out
    cdef int i, j
    for i in range(num_simps):
        for j in range(dim):
            out_view[i, j] = simps_buf[dim * i + j]

    free(simps_buf)
    return out, status
