import time

import numpy as np



def jacobi(A, b, x0, tol, nmax):
    M, N = A.shape
    L = len(x0)

    if M != N:
        print('Matrix A is not a square matrix')
        return None, 0, 0, 0
    elif L != M:
        print('Dimensions of matrix A do not match dimensions of initial guess x0')
        return None, 0, 0, 0
    if np.any(np.diag(A) == 0):
        print('At least one diagonal entry of A is zero. The method cannot proceed.')
        return None, 0, 0, 0

    D = np.diag(np.diag(A))
    B = D - A   

    xold = x0
    xnew = xold + 1
    nit = 0

    start_time = time.time()
    while np.linalg.norm(xnew - xold, np.inf) > tol and nit < nmax:
        xold = xnew
        xnew = np.linalg.solve(D, B.dot(xold) + b)
        nit += 1

    elapsed_time = time.time() - start_time
    err = np.linalg.norm(xnew - xold, np.inf)

    return xnew, nit, elapsed_time, err


# Definizione della matrice A e del vettore b
A = np.array([[4, -1, 0, 0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 3]], dtype=float)

b = np.array([15, 10, 10, 10], dtype=float)

# Valori iniziali
x0 = np.zeros_like(b)

# Tolleranza e massimo numero di iterazioni
tol = 1e-6
nmax = 100

# Eseguire il metodo di Jacobi
solution, iterations, elapsed_time, error = jacobi(A, b, x0, tol, nmax)

print("Soluzione:", solution)
print("Iterazioni:", iterations)
print("Tempo trascorso:", elapsed_time, "secondi")
print("Errore finale:", error)