import numpy as np

from Iterativi_Gradiente import gradiente

# Matrice e vettore termine noto
n = 5
A = np.diag(2 * np.ones(n)) + np.diag(-1 * np.ones(n-1), k=-1) + np.diag(-1 * np.ones(n-1), k=1)
b = np.ones(n)
x0 = np.zeros(n)

# Parametri di convergenza
tol = 1e-10
nmax = 100

# Eseguo il metodo del gradiente
xk, nit, elapsed_time, err = gradiente(A, b, x0, tol, nmax)

print("Soluzione approssimata:", xk)
print("Numero di iterazioni:", nit)
print("Tempo di esecuzione:", elapsed_time)
print("Errore finale:", err)
