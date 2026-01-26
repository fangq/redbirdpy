"""
Redbird Solver Module - Linear system solvers for FEM.

Provides:
    femsolve: Main solver interface with automatic method selection
    get_solver_info: Query available solver backends

Supported solvers:
    Direct: pardiso, umfpack, cholmod, superlu
    Iterative: blqmr, cg, cg+amg, gmres, bicgstab

Dependencies:
    - blocksolver: For BLQMR iterative solver (complex symmetric systems)
"""

__all__ = [
    "femsolve",
    "get_solver_info",
]

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab, splu
from typing import Dict, Tuple, Optional, Union, List, Any
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# =============================================================================
# Solver Backend Detection
# =============================================================================

_DIRECT_SOLVER = "superlu"  # fallback
_HAS_UMFPACK = False
_HAS_CHOLMOD = False
_HAS_AMG = False
_HAS_BLQMR = False

# Pardiso references
_pardiso_solve = None
_pardiso_factorized = None

try:
    from pypardiso import spsolve as pardiso_solve, factorized as pardiso_factorized

    _pardiso_solve = pardiso_solve
    _pardiso_factorized = pardiso_factorized
    _DIRECT_SOLVER = "pardiso"
except ImportError:
    pass

try:
    from scikits.umfpack import spsolve as umfpack_spsolve, splu as umfpack_splu

    _HAS_UMFPACK = True
    if _DIRECT_SOLVER == "superlu":
        _DIRECT_SOLVER = "umfpack"
except ImportError:
    umfpack_spsolve = None
    umfpack_splu = None

try:
    from sksparse.cholmod import cholesky as _cholmod_cholesky

    _HAS_CHOLMOD = True
except ImportError:
    _cholmod_cholesky = None

try:
    import pyamg as _pyamg

    _HAS_AMG = True
except ImportError:
    _pyamg = None

# Import blocksolver for BLQMR
try:
    from blocksolver import (
        blqmr,
        BLQMRResult,
        BLQMR_EXT,
        HAS_NUMBA,
        make_preconditioner,
        BLQMRWorkspace,
    )

    _HAS_BLQMR = True
except ImportError:
    blqmr = None
    BLQMRResult = None
    BLQMR_EXT = False
    HAS_NUMBA = False
    make_preconditioner = None
    BLQMRWorkspace = None


# =============================================================================
# Parallel Solver Helper Functions
# =============================================================================


def _solve_blqmr_batch(args):
    """
    Worker function for parallel BLQMR solving.

    This function runs in a separate process and reconstructs the sparse
    matrix from its components before solving.
    """
    (
        A_data,
        A_indices,
        A_indptr,
        A_shape,
        A_dtype,
        rhs_batch,
        tol,
        maxiter,
        use_precond,
        droptol,
        batch_id,
        start_col,
    ) = args

    # Import blqmr in worker process
    from blocksolver import blqmr as worker_blqmr

    # Reconstruct sparse matrix in worker process
    A = sparse.csc_matrix((A_data, A_indices, A_indptr), shape=A_shape, dtype=A_dtype)

    result = worker_blqmr(
        A,
        rhs_batch,
        tol=tol,
        maxiter=maxiter,
        M1=None,
        M2=None,
        x0=None,
        workspace=None,
        use_precond=use_precond,
        droptol=droptol,
    )

    x_batch = result.x if result.x.ndim > 1 else result.x.reshape(-1, 1)
    return batch_id, start_col, x_batch, result.flag, result.iter, result.relres


def _solve_iterative_column(args):
    """
    Worker function for parallel iterative solving (gmres, bicgstab, cg).

    Solves a single RHS column using the specified iterative method.
    """
    (
        A_data,
        A_indices,
        A_indptr,
        A_shape,
        A_dtype,
        rhs_col,
        col_idx,
        solver_type,
        tol,
        maxiter,
        use_amg,
    ) = args

    from scipy.sparse.linalg import gmres, bicgstab, cg
    from scipy import sparse

    # Reconstruct sparse matrix in worker process
    A = sparse.csc_matrix((A_data, A_indices, A_indptr), shape=A_shape, dtype=A_dtype)

    # Setup preconditioner if AMG requested
    M = None
    if use_amg and solver_type == "cg":
        try:
            import pyamg

            ml = pyamg.smoothed_aggregation_solver(A.tocsr())
            M = ml.aspreconditioner()
        except ImportError:
            pass

    # Select solver
    if solver_type == "gmres":
        solver_func = gmres
    elif solver_type == "bicgstab":
        solver_func = bicgstab
    elif solver_type == "cg":
        solver_func = cg
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    # Solve
    try:
        x_col, info = solver_func(A, rhs_col, M=M, rtol=tol, maxiter=maxiter)
    except TypeError:
        # Older scipy versions use 'tol' instead of 'rtol'
        x_col, info = solver_func(A, rhs_col, M=M, tol=tol, maxiter=maxiter)

    return col_idx, x_col, info


def _blqmr_parallel(
    Amat: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    tol: float,
    maxiter: int,
    rhsblock: int,
    use_precond: bool,
    droptol: float,
    nthread: int,
    verbose: bool,
) -> Tuple[np.ndarray, int]:
    """
    Solve BLQMR with multiple RHS in parallel using multiprocessing.
    """
    n, ncol = rhs.shape

    # Determine output dtype based on matrix and RHS
    is_complex = np.iscomplexobj(Amat) or np.iscomplexobj(rhs)
    out_dtype = np.complex128 if is_complex else np.float64

    # Convert matrix to CSC for consistent serialization
    Acsc = Amat.tocsc()
    A_data = Acsc.data
    A_indices = Acsc.indices
    A_indptr = Acsc.indptr
    A_shape = Acsc.shape
    A_dtype = Acsc.dtype

    # Prepare batches
    batches = []
    for batch_id, start in enumerate(range(0, ncol, rhsblock)):
        end = min(start + rhsblock, ncol)
        rhs_batch = np.ascontiguousarray(rhs[:, start:end])
        batches.append(
            (
                A_data,
                A_indices,
                A_indptr,
                A_shape,
                A_dtype,
                rhs_batch,
                tol,
                maxiter,
                use_precond,
                droptol,
                batch_id,
                start,
            )
        )

    # Solve in parallel
    x = np.zeros((n, ncol), dtype=out_dtype)
    max_flag = 0

    with ProcessPoolExecutor(max_workers=nthread) as executor:
        results = executor.map(_solve_blqmr_batch, batches)

        for batch_id, start_col, x_batch, batch_flag, niter, relres in results:
            end_col = start_col + x_batch.shape[1]
            if is_complex and not np.iscomplexobj(x_batch):
                x[:, start_col:end_col] = x_batch.astype(out_dtype)
            else:
                x[:, start_col:end_col] = x_batch
            max_flag = max(max_flag, batch_flag)

            if verbose:
                print(
                    f"blqmr [{start_col+1}:{end_col}] (worker {batch_id}): "
                    f"iter={niter}, relres={relres:.2e}, flag={batch_flag}"
                )

    return x, max_flag


def _iterative_parallel(
    Amat: sparse.spmatrix,
    rhs: np.ndarray,
    solver_type: str,
    *,
    tol: float,
    maxiter: int,
    nthread: int,
    use_amg: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Solve iterative methods (gmres, bicgstab, cg) in parallel.

    Each RHS column is solved independently in a separate process.
    """
    n, ncol = rhs.shape

    is_complex = np.iscomplexobj(Amat) or np.iscomplexobj(rhs)
    out_dtype = np.complex128 if is_complex else np.float64

    # Convert matrix to CSC for serialization
    Acsc = Amat.tocsc()
    A_data = Acsc.data
    A_indices = Acsc.indices
    A_indptr = Acsc.indptr
    A_shape = Acsc.shape
    A_dtype = Acsc.dtype

    # Prepare tasks - one per non-zero RHS column
    tasks = []
    for i in range(ncol):
        if np.any(rhs[:, i] != 0):
            rhs_col = np.ascontiguousarray(rhs[:, i])
            tasks.append(
                (
                    A_data,
                    A_indices,
                    A_indptr,
                    A_shape,
                    A_dtype,
                    rhs_col,
                    i,
                    solver_type,
                    tol,
                    maxiter,
                    use_amg,
                )
            )

    # Solve in parallel
    x = np.zeros((n, ncol), dtype=out_dtype)
    max_flag = 0

    with ProcessPoolExecutor(max_workers=nthread) as executor:
        results = executor.map(_solve_iterative_column, tasks)

        for col_idx, x_col, info in results:
            x[:, col_idx] = x_col
            max_flag = max(max_flag, info)

            if verbose:
                status = "converged" if info == 0 else f"flag={info}"
                print(f"{solver_type} [col {col_idx+1}]: {status}")

    return x, max_flag


# =============================================================================
# Main Solver Interface
# =============================================================================


def femsolve(
    Amat: sparse.spmatrix,
    rhs: Union[np.ndarray, sparse.spmatrix],
    method: str = "auto",
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """
    Solve FEM linear system A*x = b with automatic solver selection.

    Parameters
    ----------
    Amat : sparse matrix
        System matrix
    rhs : ndarray or sparse matrix
        Right-hand side (n,) or (n, m) for m simultaneous RHS
    method : str
        'auto': Automatically select best solver
        'pardiso': Intel MKL PARDISO (fastest, requires pypardiso)
        'umfpack': UMFPACK (fast, requires scikit-umfpack)
        'cholmod': CHOLMOD for SPD matrices (requires scikit-sparse)
        'direct': Best available direct solver
        'superlu': SuperLU (always available)
        'blqmr': Block QMR iterative (good for complex symmetric, multiple RHS)
        'cg': Conjugate gradient (SPD only)
        'cg+amg': CG with AMG preconditioner (SPD, requires pyamg)
        'gmres': GMRES
        'bicgstab': BiCGSTAB
    **kwargs : dict
        tol : float - convergence tolerance (default: 1e-10)
        maxiter : int - maximum iterations (default: 1000)
        rhsblock : int - block size for blqmr (default: 8)
        nthread : int - parallel workers for iterative solvers
            (default: min(ncol, cpu_count), set to 1 to disable)
            Supported by: blqmr, gmres, bicgstab, cg, cg+amg
        verbose : bool - print solver progress (default: False)
        spd : bool - True if matrix is symmetric positive definite
        M, M1, M2 : preconditioners (disables parallel for gmres/bicgstab/cg)
        x0 : initial guess
        workspace : BLQMRWorkspace for blqmr
        use_precond : bool - use automatic preconditioning for blqmr (default: True)
        droptol : float - drop tolerance for ILU preconditioner (default: 0.001)

    Returns
    -------
    x : ndarray
        Solution
    flag : int
        0 = success, >0 = solver-specific warning/error code
    """
    if sparse.issparse(rhs):
        rhs = rhs.toarray()

    rhs_was_1d = rhs.ndim == 1
    if rhs_was_1d:
        rhs = rhs.reshape(-1, 1)

    n, ncol = rhs.shape
    is_complex = np.iscomplexobj(Amat) or np.iscomplexobj(rhs)
    dtype = complex if is_complex else float
    is_spd = kwargs.get("spd", False)
    tol = kwargs.get("tol", 1e-10)
    maxiter = kwargs.get("maxiter", 1000)
    verbose = kwargs.get("verbose", False)

    x = np.zeros((n, ncol), dtype=dtype)
    flag = 0

    def get_direct_solver_for_matrix():
        """Get best direct solver for current matrix type."""
        # Pardiso is fastest and now supports complex (via real-valued formulation)
        if _DIRECT_SOLVER == "pardiso":
            return "pardiso"
        # For complex matrices without Pardiso, prefer UMFPACK
        if is_complex:
            return "umfpack" if _HAS_UMFPACK else "superlu"
        # For real matrices, use best available
        return _DIRECT_SOLVER

    # Auto-select solver
    if method == "auto":
        if n < 10000:
            method = "direct"
        elif is_spd and _HAS_AMG and not is_complex:
            method = "cg+amg"
        else:
            method = "direct"

    if method == "direct":
        method = get_direct_solver_for_matrix()

    if verbose:
        print(f"femsolve: method={method}, n={n}, ncol={ncol}, complex={is_complex}")

    # === DIRECT SOLVERS ===

    if method == "pardiso":
        if _DIRECT_SOLVER != "pardiso":
            warnings.warn("pypardiso not available, falling back")
            return femsolve(Amat, rhs, method="direct", **kwargs)

        if is_complex:
            # Convert complex system to real-valued form:
            # [A_r  -A_i] [x_r]   [b_r]
            # [A_i   A_r] [x_i] = [b_i]
            A_r = Amat.real
            A_i = Amat.imag

            # Build block matrix [A_r, -A_i; A_i, A_r]
            if sparse.issparse(Amat):
                A_real = sparse.bmat([[A_r, -A_i], [A_i, A_r]], format="csr")
            else:
                A_real = np.block([[A_r, -A_i], [A_i, A_r]])

            # Stack RHS: [b_r; b_i]
            rhs_real = np.vstack([rhs.real, rhs.imag])

            # Solve real system (batch solve for all RHS at once)
            x_real = _pardiso_solve(A_real, rhs_real)

            # Reconstruct complex solution: x = x_r + j*x_i
            x = x_real[:n, :] + 1j * x_real[n:, :]
        else:
            # Real matrix - batch solve all RHS at once
            Acsr = Amat.tocsr()
            x = _pardiso_solve(Acsr, rhs)

    elif method == "umfpack":
        if not _HAS_UMFPACK:
            warnings.warn("scikit-umfpack not available, falling back to superlu")
            return femsolve(Amat, rhs, method="superlu", **kwargs)

        Acsc = Amat.tocsc()
        # Use UMFPACK via scipy's spsolve (auto-selects UMFPACK when installed)
        # spsolve can handle matrix RHS directly
        if ncol > 1:
            # Check for zero columns and solve non-zero ones
            nonzero_cols = [i for i in range(ncol) if np.any(rhs[:, i] != 0)]
            if len(nonzero_cols) == ncol:
                # All columns non-zero - solve all at once
                x[:] = spsolve(Acsc, rhs)
            else:
                # Some zero columns - solve only non-zero ones
                for i in nonzero_cols:
                    x[:, i] = spsolve(Acsc, rhs[:, i])
        else:
            x[:, 0] = spsolve(Acsc, rhs[:, 0])

    elif method == "cholmod":
        if not _HAS_CHOLMOD:
            warnings.warn("scikit-sparse not available, falling back")
            return femsolve(Amat, rhs, method="direct", **kwargs)

        if is_complex:
            fallback = get_direct_solver_for_matrix()
            if verbose:
                print(f"cholmod doesn't support complex, using {fallback}")
            return femsolve(Amat, rhs, method=fallback, **kwargs)

        if not is_spd:
            warnings.warn("cholmod requires SPD matrix, falling back")
            return femsolve(Amat, rhs, method="direct", **kwargs)

        Acsc = Amat.tocsc()
        factor = _cholmod_cholesky(Acsc)
        for i in range(ncol):
            if np.any(rhs[:, i] != 0):
                x[:, i] = factor(rhs[:, i])

    elif method == "superlu":
        Acsc = Amat.tocsc()
        # For multiple RHS: factorize once, then solve
        # lu.solve() can handle 2D arrays directly
        if ncol > 1:
            try:
                lu = splu(Acsc)
                # Check for zero columns
                nonzero_cols = [i for i in range(ncol) if np.any(rhs[:, i] != 0)]
                if len(nonzero_cols) == ncol:
                    # All columns non-zero - solve all at once
                    x[:] = lu.solve(rhs)
                else:
                    # Some zero columns - solve only non-zero ones
                    if len(nonzero_cols) > 0:
                        rhs_nonzero = rhs[:, nonzero_cols]
                        x_nonzero = lu.solve(rhs_nonzero)
                        for idx, col in enumerate(nonzero_cols):
                            x[:, col] = (
                                x_nonzero[:, idx] if x_nonzero.ndim > 1 else x_nonzero
                            )
            except Exception:
                # Fallback to individual solves if batch fails
                for i in range(ncol):
                    if np.any(rhs[:, i] != 0):
                        x[:, i] = spsolve(Acsc, rhs[:, i])
        else:
            x[:, 0] = spsolve(Acsc, rhs[:, 0])

    # === ITERATIVE SOLVERS ===

    elif method == "blqmr":
        if not _HAS_BLQMR:
            warnings.warn("blocksolver not available, falling back to gmres")
            return femsolve(Amat, rhs, method="gmres", **kwargs)

        M1 = kwargs.get("M1", None)
        M2 = kwargs.get("M2", None)
        x0 = kwargs.get("x0", None)
        rhsblock = kwargs.get("rhsblock", 8)
        workspace = kwargs.get("workspace", None)
        use_precond = kwargs.get("use_precond", True)
        droptol = kwargs.get("droptol", 0.001)
        nthread = kwargs.get("nthread", None)
        if nthread is None:
            nthread = min(ncol, multiprocessing.cpu_count())

        if rhsblock <= 0 or ncol <= rhsblock:
            # Single batch - no parallelization needed
            result = blqmr(
                Amat,
                rhs,
                tol=tol,
                maxiter=maxiter,
                M1=M1,
                M2=M2,
                x0=x0,
                workspace=workspace,
                use_precond=use_precond,
                droptol=droptol,
            )
            x = result.x if result.x.ndim > 1 else result.x.reshape(-1, 1)
            flag = result.flag
            if verbose:
                print(
                    f"blqmr: iter={result.iter}, relres={result.relres:.2e}, "
                    f"flag={flag}, BLQMR_EXT={BLQMR_EXT}"
                )
        elif nthread > 1:
            # Parallel batch solving using multiprocessing
            x, flag = _blqmr_parallel(
                Amat,
                rhs,
                tol=tol,
                maxiter=maxiter,
                rhsblock=rhsblock,
                use_precond=use_precond,
                droptol=droptol,
                nthread=nthread,
                verbose=verbose,
            )
        else:
            # Sequential batch solving
            max_flag = 0
            for start in range(0, ncol, rhsblock):
                end = min(start + rhsblock, ncol)
                rhs_batch = rhs[:, start:end]
                x0_batch = x0[:, start:end] if x0 is not None else None

                result = blqmr(
                    Amat,
                    rhs_batch,
                    tol=tol,
                    maxiter=maxiter,
                    M1=M1,
                    M2=M2,
                    x0=x0_batch,
                    workspace=workspace,
                    use_precond=use_precond,
                    droptol=droptol,
                )
                x_batch = result.x if result.x.ndim > 1 else result.x.reshape(-1, 1)
                x[:, start:end] = x_batch
                max_flag = max(max_flag, result.flag)

                if verbose:
                    print(
                        f"blqmr [{start+1}:{end}]: iter={result.iter}, relres={result.relres:.2e}"
                    )
            flag = max_flag

    elif method == "cg+amg":
        if not _HAS_AMG:
            warnings.warn("pyamg not available, falling back to CG")
            return femsolve(Amat, rhs, method="cg", **kwargs)

        if is_complex:
            warnings.warn("cg+amg doesn't support complex, falling back to gmres")
            return femsolve(Amat, rhs, method="gmres", **kwargs)

        nthread = kwargs.get("nthread", None)
        if nthread is None:
            nthread = min(ncol, multiprocessing.cpu_count())

        if nthread > 1 and ncol > 1:
            # Parallel solving
            x, flag = _iterative_parallel(
                Amat,
                rhs,
                "cg",
                tol=tol,
                maxiter=maxiter,
                nthread=nthread,
                use_amg=True,
                verbose=verbose,
            )
        else:
            # Sequential solving
            ml = _pyamg.smoothed_aggregation_solver(Amat.tocsr())
            M = ml.aspreconditioner()

            for i in range(ncol):
                if np.any(rhs[:, i] != 0):
                    try:
                        x[:, i], info = cg(
                            Amat, rhs[:, i], M=M, rtol=tol, maxiter=maxiter
                        )
                    except TypeError:
                        x[:, i], info = cg(
                            Amat, rhs[:, i], M=M, tol=tol, maxiter=maxiter
                        )
                    flag = max(flag, info)
                    if verbose:
                        status = "converged" if info == 0 else f"flag={info}"
                        print(f"cg+amg [col {i+1}]: {status}")

    elif method == "cg":
        if is_complex:
            warnings.warn("cg requires Hermitian matrix, falling back to gmres")
            return femsolve(Amat, rhs, method="gmres", **kwargs)

        nthread = kwargs.get("nthread", None)
        if nthread is None:
            nthread = min(ncol, multiprocessing.cpu_count())
        M = kwargs.get("M", None)

        if nthread > 1 and ncol > 1 and M is None:
            # Parallel solving (without custom preconditioner)
            x, flag = _iterative_parallel(
                Amat,
                rhs,
                "cg",
                tol=tol,
                maxiter=maxiter,
                nthread=nthread,
                use_amg=False,
                verbose=verbose,
            )
        else:
            # Sequential solving
            for i in range(ncol):
                if np.any(rhs[:, i] != 0):
                    try:
                        x[:, i], info = cg(
                            Amat, rhs[:, i], M=M, rtol=tol, maxiter=maxiter
                        )
                    except TypeError:
                        x[:, i], info = cg(
                            Amat, rhs[:, i], M=M, tol=tol, maxiter=maxiter
                        )
                    flag = max(flag, info)
                    if verbose:
                        status = "converged" if info == 0 else f"flag={info}"
                        print(f"cg [col {i+1}]: {status}")

    elif method == "gmres":
        nthread = kwargs.get("nthread", None)
        if nthread is None:
            nthread = min(ncol, multiprocessing.cpu_count())
        M = kwargs.get("M", None)

        if nthread > 1 and ncol > 1 and M is None:
            # Parallel solving (without custom preconditioner)
            x, flag = _iterative_parallel(
                Amat,
                rhs,
                "gmres",
                tol=tol,
                maxiter=maxiter,
                nthread=nthread,
                use_amg=False,
                verbose=verbose,
            )
        else:
            # Sequential solving
            for i in range(ncol):
                if np.any(rhs[:, i] != 0):
                    try:
                        x[:, i], info = gmres(
                            Amat, rhs[:, i], M=M, rtol=tol, maxiter=maxiter
                        )
                    except TypeError:
                        x[:, i], info = gmres(
                            Amat, rhs[:, i], M=M, tol=tol, maxiter=maxiter
                        )
                    flag = max(flag, info)
                    if verbose:
                        status = "converged" if info == 0 else f"flag={info}"
                        print(f"gmres [col {i+1}]: {status}")

    elif method == "bicgstab":
        nthread = kwargs.get("nthread", None)
        if nthread is None:
            nthread = min(ncol, multiprocessing.cpu_count())
        M = kwargs.get("M", None)

        if nthread > 1 and ncol > 1 and M is None:
            # Parallel solving (without custom preconditioner)
            x, flag = _iterative_parallel(
                Amat,
                rhs,
                "bicgstab",
                tol=tol,
                maxiter=maxiter,
                nthread=nthread,
                use_amg=False,
                verbose=verbose,
            )
        else:
            # Sequential solving
            for i in range(ncol):
                if np.any(rhs[:, i] != 0):
                    try:
                        x[:, i], info = bicgstab(
                            Amat, rhs[:, i], M=M, rtol=tol, maxiter=maxiter
                        )
                    except TypeError:
                        x[:, i], info = bicgstab(
                            Amat, rhs[:, i], M=M, tol=tol, maxiter=maxiter
                        )
                    flag = max(flag, info)
                    if verbose:
                        status = "converged" if info == 0 else f"flag={info}"
                        print(f"bicgstab [col {i+1}]: {status}")

    else:
        raise ValueError(f"Unknown solver: {method}")

    # Flatten output if input was 1D
    if rhs_was_1d:
        x = x.ravel()

    return x, flag


def get_solver_info() -> dict:
    """Return information about available solvers."""
    info = {
        "direct_solver": _DIRECT_SOLVER,
        "has_pardiso": _DIRECT_SOLVER == "pardiso",
        "has_umfpack": _HAS_UMFPACK,
        "has_cholmod": _HAS_CHOLMOD,
        "has_amg": _HAS_AMG,
        "has_blqmr": _HAS_BLQMR,
        "complex_direct": "umfpack" if _HAS_UMFPACK else "superlu",
        "complex_iterative": ["gmres", "bicgstab"],
        "cpu_count": multiprocessing.cpu_count(),
    }

    if _HAS_BLQMR:
        info["blqmr_backend"] = "fortran" if BLQMR_EXT else "native"
        info["blqmr_has_numba"] = HAS_NUMBA
        info["complex_iterative"].insert(0, "blqmr")

    return info
