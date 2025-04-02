import numpy as np
import time

def potenze(A, q0, tol, nmax):
    global nu, qnew
    nit = 0
    err = 1
    qold = q0

    start_time = time.time()
    while err > tol and nit < nmax:
        z = A @ qold
        qnew = z / np.linalg.norm(z) # normalizzo vettore
        nu = np.conj(qnew) @ A @ qnew
        err = (qnew - qold) / np.linalg.norm(qnew)
        nit += 1
        qold = qnew

    elapsed_time = time.time() - start_time

    return qnew, nu, elapsed_time, err
