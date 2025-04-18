import numpy as np
from scipy.integrate import quad
from Cavalieri_Simpson_composito import cavalieri_simpson_composito

# returs Fourier series coefficients up to order n
# INPUT f function
#       p period
#       n order of the Fourier series
# OUTPUT coefficients of the truncated Fourier series
#        first  the coefficient  of the constant function
#        second the coefficients of k   cosine   functions
#        third  the coefficients of k   sine     functions

def fourier_series(f, p, n):

    coefficients = np.zeros(1 + 2 * n)

    # Coefficiente della componente costante (a0)
    integral_const, _ = quad(f, 0, p, limlst=100)
    coefficients[0] = (1/p) * integral_const
    # coefficients[0] = (1/p) * cavalieri_simpson_composito(0, p, f, 50)

    # Coefficienti delle funzioni coseno (a_k)
    for j in range(1, n + 1):
        f_cos = lambda x: f(x) * np.cos(j * (2 * np.pi / p) * x)
        integral_cos, _ = quad(f_cos, 0, p, limlst=100)
        coefficients[j] = (2 / p) * integral_cos

    # Coefficienti delle funzioni seno (b_k)
    for j in range(1, n + 1):
        f_sin = lambda x: f(x) * np.sin(j * (2 * np.pi / p) * x)
        integral_sin, _ = quad(f_sin, 0, p, limlst=100)
        coefficients[n + j] = (2 / p) * integral_sin

    return coefficients




