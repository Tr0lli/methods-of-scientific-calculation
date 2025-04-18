import matplotlib
matplotlib.use('Agg')  # Forza l'uso del backend 'Agg' (non interattivo)

import numpy as np
import matplotlib.pyplot as plt
from Fourier_series import fourier_series

# Parametri
n = 100  # Ordine della serie di Fourier
# Definizione della funzione e del periodo
f = lambda x: np.sin(x)
p = 2 * np.pi
# f = lambda x: x * (1 - x)
# p = 1
#f = lambda x: x
#p = 1
# f = lambda x: x * (1 - x) * np.exp(10 * x)
# p = 1
# f = lambda x: np.sign(x - 0.5)
# p = 1

# Calcolo dei coefficienti della serie di Fourier
coefficients = fourier_series(f, p, n) #, use_quad=True Puoi cambiare use_quad=False per usare Cavalieri-Simpson composito

# Generazione del vettore x
x_vect = np.arange(0, p, 0.001)
y_vect_fun = f(x_vect)

# Inizializzazione del vettore per la serie di Fourier
y_vect_Fourier = np.zeros_like(x_vect)

# Termine costante
y_vect_Fourier += coefficients[0]

# Contributo dei coseni
for j in range(1, n + 1):
    y_vect_Fourier += coefficients[j] * np.cos(j * (2 * np.pi / p) * x_vect)

# Contributo dei seni
for j in range(1, n + 1):
    y_vect_Fourier += coefficients[n + j] * np.sin(j * (2 * np.pi / p) * x_vect)

# Plot dei risultati
plt.figure()
plt.plot(x_vect, y_vect_fun, 'r*', label='Funzione esatta')
plt.plot(x_vect, y_vect_Fourier, 'b*-', label='Serie di Fourier troncata')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Confronto tra funzione e la sua serie di Fourier troncata')
plt.legend()
plt.grid(True)
plt.savefig('fourier_comparison.png')  # Salva la figura come immagine
# plt.show() # Commenta o rimuovi plt.show() quando usi il backend 'Agg'