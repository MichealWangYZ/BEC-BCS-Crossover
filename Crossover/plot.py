import matplotlib.pyplot as plt
import numpy as np
from math import *

tau = 1
beta = 1  # inverse temperature
mu = 0.5  # chemical potential

v = 3  # The BEC limit
q = np.linspace(0, 10, 100, endpoint=False)

"""
def z_c(mu, q):
    return 2*(mu-q**2/4)


def z_p(mu, q, v):
    return z_c(mu, q) + 2*v**2


def x(z):
    return z/(1-z)


def f(z, q):
    return 8*sqrt(2/tau)*exp(z_c(mu, q)*tau)*exp(-x(z)**2)*x(z)**2*exp(beta*(x(z)**2/tau-z_c(mu, q)))/(1-exp(beta*(x(z)**2/tau-z_c(mu, q))))/(x(z)**2+2*tau*v**2)
"""

y = (16*pi*v*np.exp(2*(mu-q**2/4)*tau)/(np.exp(beta*2*(mu-q**2/4))-1)-4*pi/(v-np.sqrt(q**2/4-mu)/beta))
plt.plot(q, y)
plt.show()
