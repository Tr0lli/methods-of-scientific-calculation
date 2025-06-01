import time

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from pagerank import pagerank

def prtest_one(n, rown, gamma):
    perc = rown / n
    H = sp.rand(n, n, density=perc, format='csr')
    H.data[:] = 1  # rende binaria la matrice

    start_time = time.time()
    y = pagerank(H, gamma)
    elapsed = time.time() - start_time
    print("Elapsed time:", elapsed)

    Y = np.sort(y)
    I = np.argsort(y)
    return Y, I

def prtest_two(n, rown, gamma):
    perc = rown / n
    e = np.ones(n)
    H = sp.rand(n, n, density=perc, format='csr')
    H.data[:] = 1  # matrice binaria

    outlink = H @ e
    inlink = (e @ H).T

    y = pagerank(H, gamma)
    Y = np.sort(y)
    I = np.argsort(y)

    # Pagerank vs outlink
    plt.figure(1)
    plt.plot(outlink[I], Y, '.')
    P1 = np.polyfit(outlink[I], Y, 1)
    X = np.array([0, np.max(outlink[I])])
    PP1 = np.polyval(P1, X)
    plt.plot(X, PP1, 'r')
    plt.title("Pagerank vs Outlink")

    # Pagerank vs inlink
    plt.figure(2)
    plt.plot(inlink[I], Y, '.')
    P2 = np.polyfit(inlink[I], Y, 1)
    X = np.array([0, np.max(inlink[I])])
    PP2 = np.polyval(P2, X)
    plt.plot(X, PP2, 'r')
    plt.title("Pagerank vs Inlink")

    # Pendenza della retta scalata
    xscal = np.mean(inlink[I])
    yscal = np.mean(Y)
    pend = xscal * P2[0] / yscal
    print("Pendenza retta di regressione scalata:", pend)

    plt.show()
