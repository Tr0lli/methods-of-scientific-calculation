import numpy as np

def cavalieri_simpson(a, b, f):
    """
    Calcola l'integrale approssimato di una funzione f sull'intervallo [a, b]
    utilizzando la regola di Cavalieri-Simpson.

    Args:
      a: Limite inferiore dell'intervallo.
      b: Limite superiore dell'intervallo.
      f: La funzione da integrare (deve accettare un singolo argomento numerico
         o un array NumPy e restituire il valore corrispondente o un array).

    Returns:
      L'approssimazione dell'integrale.
    """
    I = ((b - a) / 6) * (f(a) + 4 * f((a + b) / 2) + f(b))
    return I

# Esempio di utilizzo:
def my_function(x):
    return x**2

lower_bound = 0
upper_bound = 2

integral_approximation = cavalieri_simpson(lower_bound, upper_bound, my_function)
print(f"L'approssimazione dell'integrale è: {integral_approximation}")