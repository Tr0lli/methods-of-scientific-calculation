import numpy as np

def compare_dct(N, f):
    fj = []
    for j in range(0, N-1):
        fj[j] = f((2*j +1)/(2*N))

    c = dct(N, fj)
    c_troncato = c[0:N/2]

    f_new = idct(N, c_troncato)

    return c_troncato, f_new



def dct(N, fj):
    c = []
    for k in range(0, N-1):
        if k == 0:
            alpha = 1/np.sqrt(N)
        else:
            alpha = np.sqrt(2/N)
        for j in range(0, N-1):
            c[k] = alpha * (fj[j] * np.cos(k*np.pi*((2*j+1)/(2*N))))
    return c

def idct(N, c):
    fj = []
    for j in range(0, N-1):
        for k in range(0, N-1):
            if k == 0:
                alpha = 1/np.sqrt(N)
            else:
                alpha = np.sqrt(2/N)

            fj[j] = c[k] * alpha * np.cos(k*np.pi*((2*j+1)/(2*N)))
    return fj