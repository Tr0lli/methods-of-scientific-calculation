import numpy as np

def triang_inf(L, b):
    M, N = L.shape
    x = np.zeros(M)

    if M != N:
        print('Matrix L is not a square matrix')
    elif not np.allclose(L, np.tril(L)):
        print('Matrix L is not a lower triangular matrix')
    else:
        x[0] = b[0] / L[0, 0]
        for i in range(1, N):
            x[i] = (b[i] - np.dot(L[i, 0:i], x[0:i])) / L[i, i]

    return x

def triang_sup(U, b):
    M, N = U.shape
    x = np.zeros(N)

    if M != N:
        print('Matrix U is not a square matrix')
    elif not np.allclose(U, np.triu(U)):
        print('Matrix U is not an upper triangular matrix')
    else:
        x[N - 1] = b[N - 1] / U[N - 1, N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = (b[i] - np.dot(U[i, i + 1:N], x[i + 1:N])) / U[i, i]

    return x
