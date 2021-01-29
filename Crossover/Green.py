from scipy.integrate import quad, dblquad, tplquad
import numpy as np
import scipy
import numpy.matlib as mat
import matplotlib.pyplot as plt
import csv
from math import *
import cmath
import random as rd

# Parameters
tau = 2
beta = 1  # inverse temperature

v = 2  # The BEC limit

mu = -v**2  # chemical potential BEC limit


q_range = 100
q_inter = 0.01

k = 5



def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))

    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def complex_dblquad(func, a, b, c, d, **kwargs):
    def real_func(x, y):
        return scipy.real(func(x, y))

    def imag_func(x, y):
        return scipy.imag(func(x, y))
    real_integral = dblquad(real_func, a, b, c, d, **kwargs)
    imag_integral = dblquad(imag_func, a, b, c, d, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def complex_tplquad(func, a, b, c, d, e, f, **kwargs):
    def real_func(x, y, z):
        return scipy.real(func(x, y, z))

    def imag_func(x, y, z):
        return scipy.imag(func(x, y, z))
    real_integral = tplquad(real_func, a, b, c, d, e, f, **kwargs)
    imag_integral = tplquad(imag_func, a, b, c, d, e, f, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])


def z_c(mu, z):
    return 2*(mu-(z/(1-z))**2/4)


def z_p(mu, z, v):
    return z_c(mu, z) + 2*v**2


def n_b(x):
    return -1/(1-exp(beta*x))


def ene(x):
    return x**2-mu


def q(z):
    return z/(1-z)

def eps(z, k, x):
    return q(z)**2+k**2-2*k*q(z)*x-mu


def sigma_ana_1(z, x):
    return 2/pi*q(z)**2/(1-z)**2*(n_b(z_c(mu, z))*4*v*exp(beta*z_c(mu, z)))/(z_c(mu, z)-1j*omega_n-eps(z, k, x))


def sigma_ana_2(z, x):
    return -2/pi*q(z)**2/(1-z)**2*n_b(-eps(z, k, x))*(2*v+cmath.sqrt(-2*eps(z, k, x)-1j*omega_n-z_p(mu, z, v)))/(z_p(mu, z, v)+1j*omega_n+eps(z, k, x))


def sigma_num(z, x):
    return 2/pi**2*q(z)**2/(1-z)**2/(1-x)**2*n_b(z_c(mu, z)-x/(1-x))*cmath.sqrt(2*x/(1-x))/(x/(1-x)+2*v**2)*cmath.log((x/(1-x)-z_c(mu, z)-ene(q(z)-k)-1j*omega_n)/(x/(1-x)-z_c(mu, z)-ene(q(z)+k)-1j*omega_n))/(2*q(z)*k)*exp(z_c(mu, z)*beta-beta*x/(1-x))


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
# End Parameters

file_omega = "omega.csv"
file_G_i = "G_i.csv"
points = 25

i = 0
omega_n_arxiv = np.linspace(-40, 40, 80, endpoint=False)

omega_n_arxiv = np.sort(np.array(omega_n_arxiv)*2*pi/beta)
omega_n = omega_n_arxiv
Sig_ana = []
for omega_n in omega_n:
    Sig_ana.append(complex_dblquad(sigma_ana_1, 0, 1, -1, 1)[0] + complex_dblquad(sigma_ana_1, 0, 1, -1, 1)[0] + complex_dblquad(sigma_num, 0, 1, 0, 1)[0])

omega_n = omega_n_arxiv * 1j
Sig_ana = np.array(Sig_ana)
G_i_omega_n = 1/(omega_n-ene(k)-Sig_ana)
# plt.plot(np.imag(omega_n), np.real(G_i_omega_n))
# plt.plot(np.imag(omega_n), np.imag(G_i_omega_n))
# plt.plot(np.imag(omega_n), np.real(Sig_ana))
# plt.plot(np.imag(omega_n), np.imag(Sig_ana))


# plt.show()
G_i = np.append(np.real(G_i_omega_n), np.imag(G_i_omega_n))
append_list_as_row(file_omega, omega_n_arxiv)
append_list_as_row(file_G_i, G_i)

