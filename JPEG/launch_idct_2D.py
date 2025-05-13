import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from JPEG.dct2D import dct_2D, idct_2D
from launch_dct_2D import bar3_plot


def launcher_idct_single_basis(N=10):
    """Visualizza l'effetto dell'IDCT 2D su una singola componente di frequenza."""

    # Puoi cambiare qui il coefficiente attivato:
    c_mat = np.zeros((N, N))
    # Esempi alternativi (scommenta quello che vuoi usare):
    c_mat[-2, -2] = 1  # vicino all'alta frequenza
    # c_mat[N//2, N//2] = 1  # frequenza media
    # c_mat[1, 1] = 1  # frequenza bassa
    # c_mat[0, 0] = 1  # componente DC

    bar3_plot(c_mat, 'Coefficient matrix c (only one non-zero element)')

    f_mat = idct_2D(c_mat)
    bar3_plot(f_mat, 'Result of IDCT 2D (basis function in spatial domain)')

    plt.show()

if __name__ == '__main__':
    launcher_idct_single_basis(N=10)
