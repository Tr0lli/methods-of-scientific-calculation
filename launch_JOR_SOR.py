import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from Iterativi_JOR import jor  # Ensure jor is correctly implemented
from Iterativi_SOR import sor  # Ensure sor is correctly implemented

# Initialize parameters
n = 5
M = 5

A = np.diag(3 * np.ones(n)) - np.diag(np.ones(n-1), -1) - np.diag(np.ones(n-1), 1)
b = np.ones(n)
tol = 1e-10
nmax = 100
x0 = 0.5 * np.ones(n)

omega = 1

# Initialize result storage
x_jor_vect = np.zeros((M, n))
nit_jor_vect = np.zeros(M)
time_jor_vect = np.zeros(M)
err_jor_vect = np.zeros(M)

x_sor_vect = np.zeros((M, n))
nit_sor_vect = np.zeros(M)
time_sor_vect = np.zeros(M)
err_sor_vect = np.zeros(M)

# Iterate over M values of omega
for j in range(M):
    x_jor, nit_jor, time_jor, err_jor = jor(A, b, x0, tol, nmax, omega)
    x_sor, nit_sor, time_sor, err_sor = sor(A, b, x0, tol, nmax, omega)

    x_jor_vect[j, :] = x_jor
    nit_jor_vect[j] = nit_jor
    time_jor_vect[j] = time_jor
    err_jor_vect[j] = err_jor

    x_sor_vect[j, :] = x_sor
    nit_sor_vect[j] = nit_sor
    time_sor_vect[j] = time_sor
    err_sor_vect[j] = err_sor

    omega += 1 / M

print("x_jor_vect:", x_jor_vect)
print("nit_jor_vect:", nit_jor_vect)
print(time_jor_vect)
print(err_jor_vect)
print(x_sor_vect)
print(nit_sor_vect)
print(time_sor_vect)
print(err_sor_vect)

print(x_jor_vect.shape)
print(nit_jor_vect.shape)
