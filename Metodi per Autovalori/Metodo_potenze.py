import numpy as np
import time

from Metodi_matrici_triangolari import triang_inf, triang_sup
from Fattorizzazione_LU_pivoting import fattorizzazione_LU_pivoting

def metodo_potenze(A, q0, tol, nmax):
    nit = 0
    err = 1
    qold = q0
    nu = np.conj(qold) @ A @ qold

    start_time = time.time()
    while err > tol and nit < nmax:
        z = A @ qold
        qnew = z / np.linalg.norm(z) # normalizzo vettore
        nu = np.conj(qnew) @ A @ qnew
        err = (qnew - qold) / np.linalg.norm(qnew)
        nit += 1
        qold = qnew

    elapsed_time = time.time() - start_time

    return qold, nu, elapsed_time, err

def metodo_potenze_inv(A, q0, nmax, tol):
    nit = 0
    err = 1.0
    qold = q0

    start_time = time.time()

    # Fattorizzazione LU con pivoting
    P, L, U = fattorizzazione_LU_pivoting(A)

    while err > tol and nit < nmax:
        # Risolvi il sistema A⁻¹ * qold
        qold_perm = P @ qold
        y = triang_inf(L, qold_perm)
        z = triang_sup(U, y)

        # Normalizzazione
        qnew = z / np.linalg.norm(z)

        # Stima dell'autovalore
        nu = np.vdot(qnew, A @ qnew)  # np.vdot = qnew^H * A * qnew

        # Errore relativo (norma della differenza)
        err = np.linalg.norm(qnew - qold)
        nit += 1
        qold = qnew

    elapsed_time = time.time() - start_time

    eigenvect = qnew
    eigenvalue = nu
    return eigenvect, eigenvalue, elapsed_time, nit
