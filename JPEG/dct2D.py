import numpy as np

from JPEG.dct import compute_D


def dct_2D(f_mat):
    N = f_mat.shape[0]
    D = compute_D(N)

    # Applica DCT per colonne
    c_mat = D @ f_mat

    # Applica DCT per righe
    c_mat = (D @ c_mat.T).T

    return c_mat

def idct_2D(c_mat):
    N = len(c_mat)
    D = compute_D(N)

    # Applica IDCT per colonne
    f_mat = D.T @ c_mat

    # Applica IDCT per righe
    f_mat = (D.T @ f_mat.T).T

    return f_mat


if __name__ == "__main__":
    N = 4
    f_mat = np.random.rand(N, N)
    c_mat = dct_2D(f_mat)
    f_mat_new = idct_2D(c_mat)

    print(f_mat)
    print()
    print(c_mat)
    print()
    print(f_mat_new)
