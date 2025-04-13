import numpy as np

from Iterativi_Jacobi import jacobi
from Iterativi_Gauss_Seidel import gauss_seidel

n = 100

# Costruzione della matrice A:
# A inizialmente è una matrice diagonale con elementi 3 sulla diagonale principale.
A = np.diag(3 * np.ones(n))
# Modifichiamo A sottraendo 10*diag(ones(n-1), -1) e diag(ones(n-1), 1) per ottenere una struttura tridiagonale.
A = A - 10 * np.diag(np.ones(n-1), k=-1) - np.diag(np.ones(n-1), k=1)

# Vettore dei termini noti b
b = np.ones(n)

# Parametri di tolleranza, massimo numero di iterazioni e stima iniziale
tol = 1e-10
nmax = 100
x0 = np.random.rand(n)

# Esecuzione dei metodi
x_Jac, nit_Jac, time_Jac, err_Jac = jacobi(A, b, x0, tol, nmax)

x_GSe, nit_GSe, err_GSe = gauss_seidel(A, b, x0, tol, nmax)

print("Errore finale Jacobi:", err_Jac)
print("Errore finale Gauss-Seidel:", err_GSe)