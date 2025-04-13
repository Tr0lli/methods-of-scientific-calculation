import numpy as np

# Dimensione della matrice
n = 5

# Funzione per creare una matrice triangolare superiore casuale
def createU(n):
    return np.triu(np.random.randint(1, 10, (n, n)))  # Matrice triangolare superiore con valori tra 1 e 9

# Funzione per estrarre la diagonale della matrice
def diag(U):
    return np.diag(U)

# Creazione della matrice U
U = createU(n)

# Vettore x di uni
x = np.ones(n)

# Calcolo di b = U * x (moltiplicazione matrice-vettore)
b = np.dot(U, x)

# Funzione per risolvere il sistema triangolare superiore
def mySolve(U, b):
    n = len(b)  # Ottiene la dimensione del sistema
    x = np.zeros(n)  # Inizializza il vettore soluzione con zeri

    for i in range(n-1, -1, -1):  # Sostituzione all'indietro
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]  # Solo le colonne a destra della diagonale

    return x

# Calcolo della soluzione
xh = mySolve(U, b)

# Calcolo dell'errore relativo con la norma Euclidea
errore_relativo = np.linalg.norm(x - xh) / np.linalg.norm(x)

# Output dei risultati
print("Matrice U:\n", U)
print("Vettore x originale:", x)
print("Vettore b:", b)
print("Vettore xh calcolato:", xh)
print("Errore relativo:", errore_relativo)
