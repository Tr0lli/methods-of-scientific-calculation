import numpy as np
import time

def jor(A, b, x0, tol, nmax, omega):
    """
    Jacobi Over-Relaxation (JOR) method for solving the system Ax = b.

    Parameters:
    - A: coefficient matrix (must be square)
    - b: right-hand side vector
    - x0: initial guess
    - tol: tolerance for convergence
    - nmax: maximum number of iterations
    - omega: relaxation parameter (0 < omega <= 2)

    Returns:
    - x: approximate solution
    - nit: number of iterations
    - elapsed_time: execution time
    - err: final error
    """
    M, N = A.shape
    if M != N:
        raise ValueError("Matrix A must be square")
    if len(x0) != M:
        raise ValueError("Matrix A dimensions do not match the size of x0")

    # Decomposition A = D - B
    D = np.diag(np.diag(A))  # Extract diagonal matrix D
    B = D - A  # B is the non-diagonal part of A

    # Compute the inverse of D
    D_inv = np.linalg.inv(D)

    xold = np.array(x0, dtype=float)
    xnew = xold + 1  # Force entry into the loop
    nit = 0

    start_time = time.time()  # Start time measurement
    while np.linalg.norm(xnew - xold, np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        xnew = D_inv @ (b - B @ xold)  # Jacobi step
        xnew = omega * xnew + (1 - omega) * xold  # Over-Relaxation step
        nit += 1

    elapsed_time = time.time() - start_time  # Compute execution time
    err = np.linalg.norm(xnew - xold, np.inf)  # Compute final error

    return xnew, nit, elapsed_time, err
