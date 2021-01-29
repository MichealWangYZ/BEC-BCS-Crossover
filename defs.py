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
beta = 4  # inverse temperature

v = 0.3  # The BEC limit

mu = -v**2  # chemical potential BEC limit


q_range = 100
q_inter = 0.01

k = 5
points = 40
final_say = 32


lower_lim = -100
upper_lim = 200
sample_points = 300
omega_n_arxiv = np.linspace(lower_lim, upper_lim, sample_points, endpoint=False)*2*pi/beta


def z_c(mu, z):
    return 2*(mu-(z/(1-z))**2/4)


def z_p(mu, z, v):
    return z_c(mu, z) + 2*v**2


def n_b(x):
    return -1/(1-exp(beta*x))


def ene(x):
    return x**2-mu


def n_b_extra(x):
    return -exp(-beta * x)/(exp(-beta * x) - 1)


def n_f(x):
    return exp(- beta * x)/(1 + exp(- beta * x))


def q(z):
    return z/(1-z)


def eps(z, k, x):
    return q(z)**2+k**2-2*k*q(z)*x-mu


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
# End Parameters


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))

    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:]


def complex_dblquad(func, a, b, c, d, **kwargs):
    def real_func(x, y):
        return np.real(func(x, y))

    def imag_func(x, y):
        return np.imag(func(x, y))
    real_integral = dblquad(real_func, a, b, c, d, **kwargs)
    imag_integral = dblquad(imag_func, a, b, c, d, **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:]


def complex_tplquad(func, a, b, c, d, e, f, **kwargs):
    def real_func(x, y, z):
        return np.real(func(x, y, z))

    def imag_func(x, y, z):
        return np.imag(func(x, y, z))
    real_integral = tplquad(real_func, a, b, c, d, e, f, **kwargs)
    imag_integral = tplquad(imag_func, a, b, c, d, e, f, **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:]

