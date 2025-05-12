import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from JPEG.dct2D import dct_2D, idct_2D

# Funzioni di DCT 2D e IDCT 2D (esempi generici, usa il tuo codice specifico per queste)
"""def dct_2D(matrix):
    return np.fft.fftshift(np.fft.fft2(matrix))

def idct_2D(matrix):
    return np.fft.ifft2(np.fft.ifftshift(matrix)).real
"""

def bar3_plot(matrix, title_str, position=None):
    """Simula bar3 di MATLAB con matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    N = matrix.shape[0]
    _x = np.arange(N)
    _y = np.arange(N)
    _xx, _yy = np.meshgrid(_x, _y, indexing="ij")
    x, y = _xx.ravel(), _yy.ravel()
    z = np.zeros_like(x)
    dx = dy = 0.8 * np.ones_like(z)
    dz = matrix.ravel()
    ax.bar3d(x, y, z, dx, dy, dz, shade=True)
    ax.set_title(title_str)
    plt.tight_layout()

def generate_matrix(N, param=0.3):
    """Genera la matrice f_mat, basata sulla funzione segnata."""
    f = lambda x, y: np.sign(x - 0.5) * np.sign(y - 0.5)  # Funzione esempio
    f_mat = np.zeros((N, N))

    for j in range(N):
        for ell in range(N):
            x_val = (2 * j - 1) / (2 * N)
            y_val = (2 * ell - 1) / (2 * N)
            f_mat[j, ell] = f(x_val, y_val)

    return f_mat

def launcher(N=8, param=0.3):
    """Funzione principale per visualizzare il processo di DCT 2D."""
    f_mat = generate_matrix(N, param)

    # Visualizzazione della matrice originale
    plt.figure(1)
    bar3_plot(f_mat, 'Original bidimensional array f')
    plt.show()

    # Calcolo della DCT 2D
    c_mat = dct_2D(f_mat)
    plt.figure(2)
    bar3_plot(np.abs(c_mat), 'DCT of the original bidimensional f')
    plt.show()

    # Troncamento delle frequenze
    c_mat_reduced = c_mat.copy()
    c_mat_reduced[ceil(N * param):, ceil(N * param):] = 0

    plt.figure(3)
    bar3_plot(np.abs(c_mat_reduced), 'Truncated frequencies')
    plt.show()

    # Ricostruzione tramite IDCT 2D
    f_mat_reduced = idct_2D(c_mat_reduced)
    plt.figure(4)
    bar3_plot(f_mat_reduced, 'Bidimensional array corresponding to truncated frequencies')
    plt.show()

if __name__ == '__main__':
    N = 8  # Puoi cambiare la dimensione della matrice se necessario
    launcher(N)
