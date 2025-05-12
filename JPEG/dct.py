import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # oppure 'QtAgg', se preferisci
import matplotlib.pyplot as plt

def compare_dct(N, f):
    # 1. Valutazione della funzione f nei punti (2j+1)/(2N)
    fj = np.array([f((2 * j + 1) / (2 * N)) for j in range(N)])

    # 2. Calcolo DCT
    c = dct_1D(fj)

    # 3. Tronca i coefficienti alla metà
    c_troncato = np.copy(c)
    c_troncato[N // 2:] = 0

    # 4. Ricostruzione tramite IDCT
    f_new = idct_1D(c_troncato)

    # 5. Visualizzazione
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fj, label="Originale")
    plt.plot(f_new, label="Ricostruito (IDCT)")
    plt.legend()
    plt.title("Confronto f(x) e IDCT")

    plt.subplot(1, 2, 2)
    plt.stem(c, linefmt='C0-', markerfmt='C0o', basefmt='k-', label='DCT completa')
    plt.stem(c_troncato, linefmt='C1--', markerfmt='C1s', basefmt='k-', label='DCT troncata')
    plt.legend()
    plt.title("Coefficienti DCT")
    plt.tight_layout()
    plt.show()

    return c_troncato, f_new


def compute_D(N):
    D = np.zeros((N, N))
    alpha = np.ones(N) * np.sqrt(2 / N)
    alpha[0] = 1 / np.sqrt(N)

    for k in range(N):
        for i in range(N):
            D[k, i] = alpha[k] * np.cos((k) * np.pi * (2 * i + 1) / (2 * N))
    return D

def dct_1D(f_vect):
    N = len(f_vect)
    D = compute_D(N)
    c_vect = D @ f_vect  # prodotto matrice-vettore
    """
    plt.figure()
    plt.bar(np.arange(N), c_vect)
    plt.title("DCT Coefficients")
    plt.show()
    """
    return c_vect

def idct_1D(c_vect):
    N = len(c_vect)
    D = compute_D(N)
    f_vect = D.T @ c_vect  # inversa della DCT: trasposizione
    """
    plt.figure()
    plt.bar(np.arange(N), f_vect)
    plt.title("Ricostruzione tramite IDCT")
    plt.show()
    """
    return f_vect

# Esempio di utilizzo

if __name__ == "__main__":
    N = 32
    f = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    compare_dct(N, f)

"""
if __name__ == "__main__":
    N = 16
    f_vect = np.array([np.sin(np.pi * (2 * i + 1) / (2 * N)) for i in range(N)])
    c = dct_1D(f_vect)
    f_ricostruito = idct_1D(c)
"""