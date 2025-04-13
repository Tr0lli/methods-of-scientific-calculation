import numpy as np
import time

def gradiente(A, b, x0, tol, nmax):
    """
    Metodo del Gradiente per risolvere il sistema Ax = b.

    Parametri:
    - A: matrice dei coefficienti (deve essere simmetrica e definita positiva)
    - b: vettore dei termini noti
    - x0: stima iniziale
    - tol: tolleranza per la convergenza
    - nmax: numero massimo di iterazioni

    Restituisce:
    - xk: soluzione approssimata
    - nit: numero di iterazioni
    - elapsed_time: tempo di esecuzione
    - err: errore finale
    """
    M, N = A.shape
    if M != N:
        raise ValueError("La matrice A deve essere quadrata")
    if len(x0) != M:
        raise ValueError("Le dimensioni della matrice A non corrispondono alla dimensione di x0")

    # Verifica che A sia definita positiva
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("La matrice A non è definita positiva")

    nit = 0
    err = 1
    xold = np.array(x0, dtype=float)
    start_time = time.time()  # Inizio misurazione tempo
    while nit < nmax and err > tol:
        residual = b - A @ xold  # Aggiornamento del residuo
        alpha = (residual.T @ residual) / (residual.T @ A @ residual)  # Calcolo dello step
        xnew = xold + alpha * residual  # Nuova iterazione
        err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(xnew)  # Calcolo dell'errore relativo
        xold = xnew
        nit += 1
    elapsed_time = time.time() - start_time  # Tempo di esecuzione



    return xold, nit, elapsed_time, err
