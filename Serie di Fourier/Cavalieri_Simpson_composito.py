import numpy as np
from Cavalieri_Simpson import cavalieri_simpson

def cavalieri_simpson_composito(a, b, f, N):
    """
    Calcola l'integrale approssimato di una funzione f sull'intervallo [a, b]
    utilizzando la regola di Cavalieri-Simpson composita con N sottointervalli.

    Args:
      a: Limite inferiore dell'intervallo.
      b: Limite superiore dell'intervallo.
      f: La funzione da integrare (deve accettare un singolo argomento numerico
         o un array NumPy e restituire il valore corrispondente o un array).
      N: Il numero di sottointervalli. Il numero di nodi è N + 1.

    Returns:
      L'approssimazione dell'integrale ottenuta con la regola composita.
    """
    H = (b - a) / N
    I = 0.0  # Inizializziamo l'integrale come un float

    for j in range(N):
        a1 = a + (j * H)
        b1 = a + (j + 1) * H
        I += cavalieri_simpson(a1, b1, f)  # Riutilizziamo la funzione Cavalieri_Simpson

    return I

# Esempio di utilizzo (richiede la definizione di cavalieri_simpson come nel precedente esempio):
def my_function(x):
    return x**2

lower_bound = 0
upper_bound = 2
num_subintervals = 4

integral_approximation_composito = cavalieri_simpson_composito(lower_bound, upper_bound, my_function, num_subintervals)
print(f"L'approssimazione dell'integrale con la regola composita è: {integral_approximation_composito}")