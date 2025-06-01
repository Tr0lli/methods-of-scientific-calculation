import numpy as np

from Metodi_matrici_triangolari import triang_inf, triang_sup
from Fattorizzazione_LU_pivoting import fattorizzazione_LU_pivoting
def metodo_Gauss_pivoting_totale(A, b):
    M, N = A.shape
    x = np.zeros(N)

    if M != N:
        print('Matrix A is not a square matrix')
    else:
        # Calcola la fattorizzazione PLU
        P, L, U = fattorizzazione_LU_pivoting(A)

        # Applica la permutazione a b
        b = P @ b

        # Risolvi L * y = b e poi U * x = y
        y = triang_inf(L, b)
        x = triang_sup(U, y)

    return x
