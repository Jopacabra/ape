import numpy as np
import scipy.integrate as integrate
import plasma_interaction as pi
import time
import config
import argparse

"""
This file generates the tabulated energy loss values in the 'e_loss_tables' folder.
"""
# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--coupling", help="coupling to generate, e.g. 2.0 or 2.1")

# Get command line arguments
args = parser.parse_args()
coupling_list = np.array([float(args.coupling)])  # Coupling to use

# Use default hard coded list if not passed a coupling on the command line
if coupling_list is None:
    coupling_list = [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]

subdiv = 1
for parton in ['q', 'g']:
    print('Computing tables for parton: {}'.format(parton))
    for coupling in coupling_list:
        print('Computing tables for g={}'.format(coupling))
        config.constants.G = coupling
        ALPHAS = (coupling ** 2) / (4 * np.pi)


        ####################
        # Define integrand #
        ####################

        # Create interpolated function for delta E given T, L, & E:
        def delta_E(E, T, L):

            if parton == 'q':
                CR = 4 / 3
            else:
                CR = 3

            mu = pi.mu_DeBye(T=T, g=coupling)  # in GeV
            lamb = 1 / pi.inv_lambda(T=T, hard_pid=parton)  # in GeV
            FmGeV = 1 / 0.19732687

            # # Define the analytic dI_dx we're targeting
            # def an_dI_dx(x):
            #     return ( ((1 / FmGeV) ** 2) * (CR * ALPHAS / 4) * ((1 - x + ((x**2)/2))/x) * ((L**2 * mu**2)/lamb))

            # Define the analytic Delta E at first order
            an_delta_E_1 = (((FmGeV) ** 2) * (CR * ALPHAS / 4) * ((L ** 2 * mu ** 2) / lamb) * np.log(E / mu))

            # Define analytic Delta E at zeroth order (vacuum)
            an_delta_E_0 = ((4 * CR * ALPHAS / (3 * np.pi)) * E * np.log(E / mu))

            an_delta_E = an_delta_E_1  # an_delta_E_0 + an_delta_E_1

            # Define numerical integrand as function of q and k

            # Define numerical integrand as function of q and k
            def integrand(x):
                return lambda phi, k, q: (((FmGeV) ** 3) * (4 * CR * ALPHAS / (np.pi ** 2))
                                          * (1)  # - x + ((x ** 2) / 2))
                                          * (L / lamb) * E
                                          * ((mu ** 2) / ((q ** 2 + mu ** 2) ** 2))
                                          * ((q ** 2 * np.cos(phi) * (
                                    k ** 2 - 2 * k * q * np.cos(phi) + q ** 2) * L ** 2)
                                             / (16 * x ** 2 * E ** 2 + (
                                        (k ** 2 - 2 * k * q * np.cos(phi) + q ** 2) ** 2 * L ** 2 * ((FmGeV) ** 2)))))

            abs_err = 0.1
            rel_err = 0.1

            dI_dx_finq = lambda x: 2 * (integrate.nquad(integrand(x), [[0, np.pi], [mu, np.min(
                [2 * E * x, 2 * E * np.sqrt(x * (1 - x))])], [0, np.sqrt(3 * mu * E)]],
                                                        opts={"epsabs": abs_err, "epsrel": rel_err, "limit": subdiv})[
                0])

            x_min = 0
            x_max = 1
            t0 = time.time()
            delta_E = integrate.quad(dI_dx_finq, x_min, x_max, limit=subdiv)[0]
            tf = time.time()
            print('E={} GeV, T={} GeV, L={} fm -- time={} s'.format(E, T, L, (tf - t0)))

            return delta_E


        delta_E = np.vectorize(delta_E)  # IMPORTANT!!!

        ##############################
        # Sample Delta E Phase Space #
        ##############################

        E_points = np.logspace(-0.3010299956639812, 2, 15)  # Logarithmic, 0.5 to 100 GeV
        T_points = np.logspace(-1.3010299956639813, 0, 15)  # Logarithmic, 0.05 to 1 GeV
        L_points = np.concatenate([np.array([0.0]), np.logspace(-2, 1.4, 15)])  # Logarithmic, 0.01 to 25 fm
        #  g_points = np.array([1.8, 1.9, 2, 2.1, 2.2])

        Es, Ts, Ls = np.meshgrid(E_points, T_points, L_points, indexing='ij')

        delta_E_vals = delta_E(Es, Ts, Ls)

        np.savez('e_loss_tables/g{}_deltaE_samples_{}_{}subdiv.npz'.format(coupling, parton, subdiv), E_points=E_points,
                 T_points=T_points,
                 L_points=L_points, delta_E_vals=delta_E_vals)
