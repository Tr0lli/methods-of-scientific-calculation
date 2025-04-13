import numpy as np
import time

from Diretti_triang_Lower import triang_inf  # Function to solve Lx = b

def sor(A, b, x0, tol, nmax, omega):
    """
    Successive Overrelaxation (SOR) method for solving the system Ax = b.

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
        print('Matrix A is not a square matrix')
        return None, 0, 0, 0
    if len(x0) != M:
        print('Dimensions of matrix A do not match the size of x0')
        return None, 0, 0, 0

    # Extract required matrices
    L = np.tril(A)  # Lower triangular part of A
    B = A - L  # Remaining part of A

    xold = np.array(x0, dtype=float)
    xnew = xold + 1  # Force entry into the loop
    nit = 0

    start_time = time.time()  # Start execution time measurement
    while np.linalg.norm(xnew - xold, np.inf) > tol and nit < nmax:
        xold = xnew.copy()
        xnew = triang_inf(L, (b - np.dot(B, xold)))  # Solve Lx = (b - Bxold)
        xnew = omega * xnew + (1 - omega) * xold  # SOR relaxation step
        nit += 1

    elapsed_time = time.time() - start_time  # Compute execution time
    err = np.linalg.norm(xnew - xold, np.inf)  # Compute final error

    return xnew, nit, elapsed_time, err
