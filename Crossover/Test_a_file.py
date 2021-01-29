from scipy.integrate import quad, dblquad, tplquad
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from math import *
import cmath

tau = 2
beta = 1  # inverse temperature

v = 3  # The BEC limit

mu = -v**2  # chemical potential


q_range = 100
q_inter = 0.01

k = 0.5


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))

    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:]


def complex_dblquad(func, a, b, c, d, **kwargs):
    def real_func(x, y):
        return scipy.real(func(x, y))

    def imag_func(x, y):
        return scipy.imag(func(x, y))
    real_integral = dblquad(real_func, a, b, c, d, **kwargs)
    imag_integral = dblquad(imag_func, a, b, c, d, **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:]


def complex_tplquad(func, a, b, c, d, e, f, **kwargs):
    def real_func(x, y, z):
        return scipy.real(func(x, y, z))

    def imag_func(x, y, z):
        return scipy.imag(func(x, y, z))
    real_integral = tplquad(real_func, a, b, c, d, e, f, **kwargs)
    imag_integral = tplquad(imag_func, a, b, c, d, e, f, **kwargs)
    return real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:]


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
    return 4*pi*q(z)**2/(1-z)**2*16*pi*(n_b(z_c(mu, z))*2*v*exp(beta*z_c(mu, z)))/(z_c(mu, z)-1j*omega_n-eps(z, k, x))


def sigma_ana_2(z, x):
    return 4*pi*q(z)**2/(1-z)**2*16*pi*n_b(-eps(z, k, x))*(2*v+cmath.sqrt(-2*eps(z, k, x)-1j*omega_n-z_p(mu, z, v)))/(z_p(mu, z, v)+1j*omega_n+eps(z, k, x))


def sigma_num(x):
    return 4*pi*q(z)**2/(1-z)**2/(1-x)**2*16*pi*n_b(z_c(mu, z)-x/(1-x))/(x/(1-x)+2*v**2)*cmath.log((x/(1-x)-z_c(mu, z)-ene(q(z)-k)-1j*omega_n)/(x/(1-x)-z_c(mu, z)-ene(q(z)+k)-1j*omega_n))/(2*q(z)*k)*exp(z_c(mu, z)*beta-beta*x/(1-x))

points = 15
omega_n = np.array(np.matrix(np.linspace(0, 2*points, 2*points, endpoint=False)*2*pi/beta))


# Do a Pade Transformation

sigma = np.array(np.transpose(np.matrix(omega_n)))
Sigma = np.diag(sigma)
K_1 = np.transpose(np.matrix(omega_n**0))
for i in range(1, points):
    K_1 = np.hstack((K_1, np.transpose(np.matrix(omega_n**i))))

K_2 = -Sigma * K_1
K = np.hstack(K_1, K_2)
p_q = np.linalg.inv(K)
