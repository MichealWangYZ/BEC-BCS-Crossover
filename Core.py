from defs import *
import time




mu = -0.09
def delta_rpp_raw(z, x):
    return q(z)**2 / (1-z)**2 / (2 * pi)**2 * (n_f( ene(q(z)) ) + n_f( q(z)**2 + Q**2 - 2*Q*q(z)*x - mu ))/( -1j* omega_n + ene(q(z)) + q(z)**2 + Q**2 - 2*Q*q(z)*x - mu)

##############

def Pade_fit(omega_samples, variables, order=10): # Note the imput here must be an integer!!!
    omega_samples = list(omega_samples)
    variables = list(variables)

    # Zero_point should be eliminated to avoid unexpected occurence

    zero_point = omega_samples.index(0)
    zero = omega_samples.pop(zero_point)
    zero_point_val = variables.pop(zero_point)
    # print(zero_point_val)
    omega_samples = np.array(omega_samples)
     # In the algorithm we're gonna use we need two different Variables.
    variables_copy = variables[:]   

    # The first one serves as the real Variables matrix.
    variables.append(zero_point_val) 
   
    Variables = np.diag(variables)

    # The second one, however, needs a zero.
    variables_copy.append(0) 
    # omega_n = np.array(omega_n)

    # Similarly we need two omegas
    omega_samples_first = np.append(omega_samples, 1)

    # Revised Pade Method.
    K_1 = np.transpose(np.matrix(omega_samples_first**(-order)))
    for i in range(1, order):  # final say means ... Well, FINALLY I'm gonna have this fucking stuff
        K_1 = np.hstack((K_1, np.transpose(np.matrix(np.append(omega_samples**(i-order), 0)))))


    K_2 = -Variables * K_1
    K = np.hstack((K_1, K_2))


    y = np.transpose(np.array([variables_copy]))


    p_q, resid, rank, s = np.linalg.lstsq(K, y, rcond=None) # doesn't converge indecate the problem with K
    p = np.transpose(p_q[:order])

    q = np.transpose(p_q[order:])

    q = np.hstack((q, np.matrix([1])))


    return [p, q]


####################


# The target is select 200 points out of 100000 points
def Core_pade(Q):



    # Sampling
    sample_limit = 100000 # 100,000
    sample_set = np.linspace(-10, 10, 21)
    a = []
    for i in range(11, 51):  # Don't forget to change the number after trail!!!
        a.append(floor(10**(i/10)))
    sample_set = sorted([-item for item in a]) + list(sample_set) + a
    
    omega_sampling = np.array(sample_set) * 2 * pi / beta
    # calculate for each point
    delta_rpp_0 = []
    for omega_n in omega_sampling:
        def delta_rpp_raw(z, x):
            return q(z)**2 / (1-z)**2 / (2 * pi)**2 * (n_f( ene(q(z)) ) + n_f( q(z)**2 + Q**2 - 2*Q*q(z)*x - mu ))/( -1j* omega_n + ene(q(z)) + q(z)**2 + Q**2 - 2*Q*q(z)*x - mu)
        temp = complex_dblquad(delta_rpp_raw, 0, 1, -1, 1)[0]

        delta_rpp_0.append(temp)
    # print(delta_rpp_0)

    Core_pade = Pade_fit(1j*omega_sampling, delta_rpp_0)

    return Core_pade


#################

def delta_Sigma_0(Q, omega_present, order=10):
  
    p, q = Core_pade(Q)

    X_N = np.matrix((omega_present*1j)**0)
    for i in range(1, order):
        X_N = np.vstack((X_N, np.matrix((omega_present*1j)**i)))

    X_D = np.vstack((X_N, np.matrix((omega_present*1j)**order)))
    Num = p * X_N  # N stands for numerator
    Dnu = q * X_D  # D stands for denominator
    result = np.squeeze(np.asarray(Num/Dnu))
    return result

##################

def Core(Q):
    test_time = time.time()
    omega_input = np.linspace(-10**5, 10**5, 2*10**5, endpoint=False)
    # This gives a presicion of about 10^-5
    G_0 = 1/(1j*omega_input - ene(Q))
    delta_rpp = delta_Sigma_0(Q, omega_input)
    Gamma_0 = (2*v - np.sqrt(Q**2 - 2 * 1j * omega_input - 4 * mu))/(4 * v**2 - (Q**2 - 2*1j*omega_input - 4*mu))
    delta_gamma = - Gamma_0 * Gamma_0 * delta_rpp/(1 + Gamma_0 * delta_rpp)
    Core_part = sum(delta_gamma * G_0)
    print("--- %s seconds ---" % (time.time() - test_time))
    print(Q)
    return Core_part


start_time = time.time()
z = np.linspace(0, 1, 100, endpoint=False)
k = z/(1 - z)
sigma = []
for Q in k:
    sigma.append(Core(Q))
k = z/(1 - z)
print(sigma)
plt.plot(k, sigma)
# print(complex_quadrature(Core, 0, np.Infinity, epsrel=1.49e-6))
print("--- %s seconds ---" % (time.time() - start_time))