import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import time

# --- FUNZIONE PAGERANK ---
from pagerank import pagerank  # Assicurati che questa funzione sia definita correttamente

# --- LETTURA DATI ---
print('Reading data. Be patient! Big matrices on the way')

# Il file ha due colonne: da_nodo, a_nodo
G = np.loadtxt('web-BerkStan.txt', dtype=int)

print('Creating the matrix in sparse format')
n = 685230  # Numero di nodi dichiarato nel file
N = G.shape[0]
S = np.ones(N)

# Costruzione matrice sparsa H (adjacency matrix)
H = coo_matrix((S, (G[:, 0] - 1, G[:, 1] - 1)), shape=(n, n)).tocsr()

# --- PLOT SPY DELLA MATRICE ---
print('Depicting the SPY of the matrix H')
plt.figure(1)
plt.spy(H, markersize=0.5)
plt.title("Spy plot of matrix H")

plt.figure(2)
plt.spy(H[:10000, :10000], markersize=1)
plt.title("Spy plot of H[0:10000, 0:10000]")

# --- CALCOLO PAGERANK ---
print('Page rank solved with power method')
e = np.ones(n)
gamma = 0.5

outlink = H @ e
inlink = H.T @ e

start_time = time.time()
y = pagerank(H, gamma)
elapsed = time.time() - start_time
print(f"Pagerank computed in {elapsed:.2f} seconds")

# --- ORDINAMENTO PAGERANK ---
print('Sorting the eigenvector')
Y = np.sort(y)
I = np.argsort(y)
print('The vector I contains the desired rank')

# --- GRAFICI FACOLTATIVI ---

# # Numero di link uscenti vs pagerank
# plt.figure(3)
# plt.plot(outlink[I], Y, '.')
# P1 = np.polyfit(outlink[I], Y, 1)
# X1 = np.array([0, np.max(outlink[I])])
# plt.plot(X1, np.polyval(P1, X1), 'r')
# plt.title('Pagerank vs Outlink')
# plt.xlabel('Outlink count')
# plt.ylabel('Pagerank')

# # Numero di link entranti vs pagerank
# plt.figure(4)
# plt.plot(inlink[I], Y, '.')
# P2 = np.polyfit(inlink[I], Y, 1)
# X2 = np.array([0, np.max(inlink[I])])
# plt.plot(X2, np.polyval(P2, X2), 'r')
# plt.title('Pagerank vs Inlink')
# plt.xlabel('Inlink count')
# plt.ylabel('Pagerank')

# # Calcolo pendenza scalata
# xscal = np.mean(inlink[I])
# yscal = np.mean(Y)
# pend = xscal * P2[0] / yscal
# print('Pendenza retta di regressione scalata:', pend)

plt.show()
