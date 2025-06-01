import numpy as np
import matplotlib.pyplot as plt
from Metodo_potenze import metodo_potenze_inv

# Parametri
N = 150
tol = 1e-6
nmax = 100_000

time_potenze = {}
nit_potenze = {}

# Ciclo su dimensioni crescenti
for n in range(10, N + 1, 10):
    print(f"n = {n}")

    # Matrice tridiagonale con -1 su diagonali laterali
    A = np.diag(3 * np.ones(n)) - np.diag(np.ones(n - 1), -1) - np.diag(np.ones(n - 1), 1)

    # Vettore iniziale normalizzato
    q0 = np.ones(n)
    q0 /= np.linalg.norm(q0)

    # Metodo delle potenze inverse
    eigenvect, eigenvalue, elapsed_time, nit = metodo_potenze_inv(A, q0, nmax, tol)

    # Salva i risultati
    time_potenze[n] = elapsed_time
    nit_potenze[n] = nit

# Plot tempo
plt.figure(1)
plt.plot(list(time_potenze.keys()), list(time_potenze.values()), marker='o')
plt.title('Time elapsed')
plt.xlabel('Matrix size n')
plt.ylabel('Time (s)')
plt.grid(True)

# Plot numero di iterazioni
plt.figure(2)
plt.plot(list(nit_potenze.keys()), list(nit_potenze.values()), marker='*')
plt.title('Number of iterations')
plt.xlabel('Matrix size n')
plt.ylabel('Iterations')
plt.grid(True)

plt.show()
