import numpy as np

from Diretti_triang_Lower import triang_inf


def gauss_seidel(A, b, x0, tol, nmax):
    """
    Metodo di Gauss-Seidel per risolvere Ax = b.

    INPUT:
      - A: matrice dei coefficienti (sistema quadrato)
      - b: vettore dei termini noti
      - x0: stima iniziale
      - tol: tolleranza per il criterio di arresto
      - nmax: numero massimo di iterazioni

    OUTPUT:
      - x: soluzione approssimata
      - nit: numero di iterazioni eseguite
      - elapsed_time: tempo impiegato
      - err: errore finale (norma infinito di xnew - xold)
    """
    M, N = A.shape

    if M != N:
        print('Matrix A is not a square matrix')
        return None, 0, 0, 0
    if len(x0) != M:
        print('Dimensions of matrix A do not match dimension of initial guess x0')
        return None, 0, 0, 0

    L = np.tril(A)
    B = A - L

    xOld = x0
    xNew = xOld.copy() + 1
    nit = 0

    while np.linalg.norm(xNew - xOld, np.inf) > tol and nit < nmax:
        xOld = xNew.copy()
        # Aggiornamento: risolvi L*x = (b - B*xold)
        xNew = triang_inf(L, (b - np.dot(B, xOld)))
        nit += 1

    err = np.linalg.norm(xNew - xOld, np.inf)

    return xNew, nit, err

A = np.array([[4, -1,  0,  0],
              [-1, 4, -1,  0],
              [0, -1,  4, -1],
              [0,  0, -1,  3]], dtype=float)

b = np.array([15, 10, 10, 10], dtype=float)
x0 = np.zeros_like(b)  # stima iniziale
tol = 1e-6
nmax = 100

x, nit, err = gauss_seidel(A, b, x0, tol, nmax)

print("Soluzione x:", x)
print("Iterazioni:", nit)
print("Errore finale:", err)