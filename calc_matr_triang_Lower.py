import numpy as np

def triang_inf(L, b):
    M, N = L.shape

    if M != N:
        print('Matrix L is not a square matrix')
        return None
    if not np.allclose(L, np.tril(L), atol=1e-15):
        print('Matrix L is not a lower triangular matrix')
        return None

    x = np.zeros(M, dtype=float)

    # Sostituzione in avanti
    x[0] = b[0] / L[0, 0]
    for i in range(1, N):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x

"""
# Esempio di utilizzo
L = np.array([[2, 0, 0],
              [3, 5, 0],
              [1, -2, 4]], dtype=float)

n = L.shape

# Vettore x di uni
x = np.array([2, 0.2, -3], dtype=float)

# Calcolo di b = U * x (moltiplicazione matrice-vettore)
b = np.dot(L, x)

#b = np.array([4, 7, 3], dtype=float)

xh = triang_inf(L, b)

print("Soluzione x:", xh)

"""
