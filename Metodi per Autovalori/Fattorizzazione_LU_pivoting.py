import numpy as np

def fattorizzazione_LU_pivoting(A):
    A_old = A.copy()
    N = A.shape[0]
    M = np.eye(N)
    P = np.eye(N)

    for n in range(N - 1):
        # Verifica se tutti gli elementi sotto la diagonale in colonna n sono (quasi) nulli
        if np.sum(np.abs(A_old[n + 1:, n])) < 1e-14:
            A_new = A_old  # Nessuna modifica
        else:
            # Trova la posizione del massimo in valore assoluto nella colonna n (da riga n+1 in poi)
            pos = np.argmax(np.abs(A_old[n + 1:, n])) + n + 1

            # Crea matrice di permutazione Pn che scambia riga n+1 con riga pos
            Pn = np.eye(N)
            if pos != n + 1:
                Pn[[n + 1, pos]] = Pn[[pos, n + 1]]

            # Applica permutazione
            P = Pn @ P
            A_old = Pn @ A_old

            # Costruisce Mn e Mn_inv
            Mn = np.eye(N)
            Mn[n + 1:, n] = -A_old[n + 1:, n] / A_old[n, n]

            Mn_inv = np.eye(N)
            Mn_inv[n + 1:, n] = A_old[n + 1:, n] / A_old[n, n]

            # Aggiorna M e A
            M = Mn @ Pn @ M
            A_old = Mn @ A_old

    U = A_old
    # Calcola L = P @ inv(M)
    L = P @ np.linalg.inv(M)
    return P, L, U
