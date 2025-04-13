import numpy as np
print(np.__version__)



def calc_matr_triangolareUp():
    U = np.array([[1,2,3],
                  [0,5,6],
                  [0,0,9]])

    x = np.zeros(3)

    b = U*x

    for i in range(3-1, -1, -1):
        sum_ax = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (b[i] - sum_ax) / U[i, i]

    print("Soluzione x:", x)

    




    # Soluzione in classe

    ###
    # n = 5
    # createU
    # funzione diag che prende gli elementi della diagonale della matrice
    # vettore x di uni
    # b = U*x
    #
    # xh = mySolve(U, b)
    # for i=n-1: -1:1
    #   x(i) = (b(i) - dot(U(i,:),x))/U(i,i); dot è il prodotto scalare
    # poi calcolo norma per l'errore relativo
    #
    ###




