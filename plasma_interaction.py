import numpy as np
import config
import hard_particles
import plasma
import utilities
import logging

from utilities import zeta

class HadronGas(Exception):
    """
    Raise to end evolution while computing an interaction if there is no event data for the coordinates of the particle
    """

# Function to return DeBye mass at a particular point
# Ref - https://inspirehep.net/literature/1725162
def mu_DeBye(T, g=None):
    Nf = 2  # Number of light quark flavors
    if g is None:
        g = config.constants.G
    debye_mass = g * T * np.sqrt(1 + Nf /6)
    return debye_mass


# Function to return total cross section at a particular point for parton and *gluon* in medium
# Total GW cross section, as per Sievert, Yoon, et. al.
# https://inspirehep.net/literature/1725162
def sigma(temp, hard_pid=21, soft_pid=21):
    """
    We select the appropriate cross-section for a known parton and
    known medium parton specified when called
    """
    coupling = config.constants.G

    sigma_gg_gg = (9/(32 * np.pi)) * coupling ** 4 / (mu_DeBye(T=temp) ** 2)
    sigma_qg_qg = (1/(8 * np.pi)) * coupling ** 4 / (mu_DeBye(T=temp) ** 2)
    sigma_qq_qq = (1/(18 * np.pi)) * coupling ** 4 / (mu_DeBye(T=temp) ** 2)

    if hard_pid == 21 and soft_pid == 21:
        # gg -> gg cross-section
        cross_section = sigma_gg_gg
    elif np.abs(hard_pid) < 7  and soft_pid == 21:
        # qg -> qg cross-section
        cross_section = sigma_qg_qg
    elif hard_pid == 21 and np.abs(soft_pid) < 7:
        # qg -> qg cross-section
        cross_section = sigma_qg_qg
    elif np.abs(hard_pid) < 7 and np.abs(soft_pid) < 7:
        # qq -> qq cross-section
        cross_section = sigma_qq_qq
    elif hard_pid in [22, 23, 24]:
        logging.debug('EW Boson... Using zero for scattering cross section')
        cross_section = 0
    else:
        logging.debug('Unknown particles... Using zero for scattering cross section')
        logging.debug("hard_pid:{}, soft_pid:{}".format(hard_pid, soft_pid))
        logging.debug("hard_pid == 21: {}".format(hard_pid == 21))
        logging.debug("np.abs(soft_pid) < 7: {}".format(np.abs(soft_pid) < 7))
        cross_section = 0

    return cross_section

# Function to return partial density at a particular point for given medium partons
# Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.
def rho(temp, soft_pid=21):
    if soft_pid == 21:
        density = 1.202056903159594 * 16 * (1 / (np.pi ** 2)) * temp ** 3
    elif np.abs(soft_pid) < 7:
        NDOF = 6  # 24 / 4  --> per pdg id, instead of for u, ubar, d, dbar together.
        density = 1.202056903159594 * (3/4) * NDOF * (1 / (np.pi ** 2)) * temp ** 3
    else:
        # Return 0
        density = 0
    return density

# Function to return gradient of partial density at a particular point for given medium partons
# Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.
def gradrho(temp, gradtemp, soft_pid=21):
    if soft_pid == 21:
        density = 1.202056903159594 * 16 * (1 / (np.pi ** 2)) * 3 * (temp ** 2) * gradtemp
    elif np.abs(soft_pid) < 7:
        NDOF = 6  # 24 / 4  --> per pdg id, instead of for u, ubar, d, dbar together.
        density = 1.202056903159594 * (3/4) * NDOF * (1 / (np.pi ** 2)) * 3 * (temp ** 2) * gradtemp
    else:
        # Return 0
        density = 0
    return density

# Function to return inverse QGP drift mean free path in units of GeV
# Total GW cross section, as per Sievert, Yoon, et. al.
def inv_lambda(T, hard_pid=21, soft_pid=None):
    """
    By default:
    We apply a reciprocal summation between the cross-section times density for medium gluons and for medium quarks
    to get the mean free path as in https://inspirehep.net/literature/1725162
    """

    if soft_pid is None:
        # Assumes two light quark flavors in the medium, including their antiparticles
        return (sigma(temp=T, hard_pid=hard_pid, soft_pid=21) * rho(temp=T, soft_pid=21)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=1) * rho(temp=T, soft_pid=1)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=-1) * rho(temp=T, soft_pid=-1)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=2) * rho(temp=T, soft_pid=2)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=-2) * rho(temp=T, soft_pid=-2))
    else:
        # Gives just the density of the pid you asked for.
        return sigma(temp=T, hard_pid=hard_pid, soft_pid=soft_pid) * rho(temp=T, soft_pid=soft_pid)

# Function to return inverse QGP drift mean free path in units of GeV
# Total GW cross section, as per Sievert, Yoon, et. al.
def inv_lambda_rhograd(T, gradtemp, tau, hard_pid=21, soft_pid=None):
    """
    By default:
    We apply a reciprocal summation between the cross-section times density for medium gluons and for medium quarks
    to get the mean free path as in https://inspirehep.net/literature/1725162
    """

    if soft_pid is None:
        # Assumes two light quark flavors in the medium, including their antiparticles
        return (sigma(temp=T, hard_pid=hard_pid, soft_pid=21) * (rho(temp=T, soft_pid=21) + gradrho(temp=T, gradtemp=gradtemp, soft_pid=21)*tau)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=1) * (rho(temp=T, soft_pid=1) + gradrho(temp=T, gradtemp=gradtemp, soft_pid=1)*tau)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=-1) * (rho(temp=T, soft_pid=-1) + gradrho(temp=T, gradtemp=gradtemp, soft_pid=-1)*tau)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=2) * (rho(temp=T, soft_pid=2) + gradrho(temp=T, gradtemp=gradtemp, soft_pid=2)*tau)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=-2) * (rho(temp=T, soft_pid=-2)) + gradrho(temp=T, gradtemp=gradtemp, soft_pid=-2)*tau)
    else:
        # Gives just the density of the pid you asked for.
        return sigma(temp=T, hard_pid=hard_pid, soft_pid=soft_pid) * rho(temp=T, soft_pid=soft_pid)

# Define integrand for mean q_drift (k=0 moment)
def drift_integrand(T, u_perp, u_par, E, hard_pid=21):
    FmGeV = 1/0.19732687

    # Compute inverse mfp and debye mass
    inv_lambda_val = inv_lambda(T, hard_pid=hard_pid, soft_pid=None)
    mu = mu_DeBye(T)

    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return ((FmGeV) * (1 / E) * config.jet.K_F_DRIFT
            * (3 * np.log(E/mu)
               * (u_perp / (1 - u_par))
               * (mu**2)
               * inv_lambda_val))


# Integrand for mean drift including linear gradients
def drift_integrand_linear_gradients(T, u_perp, u_par, gradtemp, tau, E, hard_pid=21):
    """
    Note that as of now we're including ONLY linear gradients of density in the presence of flow. We neglect gradients
    of the debye mass and gradients of the flow.
    """
    FmGeV = 1/0.19732687

    # Compute dot product of u_perp and gradtemp
    dotted_ugradperp = np.dot(u_perp, gradtemp)
    gradtemp_mag = dotted_ugradperp / np.linalg.norm(u_perp)

    # Compute inverse mfp and debye mass
    inv_lambda_val = inv_lambda(T, hard_pid=hard_pid, soft_pid=None)
    inv_lambda_grad_val = inv_lambda_rhograd(T, gradtemp=gradtemp_mag, tau=tau, hard_pid=hard_pid, soft_pid=None)
    mu = mu_DeBye(T)

    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return ((FmGeV) * (1 / E) * config.jet.K_F_DRIFT
            * (3 * np.log(E / mu)
               * (u_perp / (1 - u_par))
               * (mu ** 2)
               * (inv_lambda_val - np.linalg.norm(u_perp)*inv_lambda_grad_val/(1-u_par))))
                # Note the minus sign in front of the last term. This makes the term constructive.




# Integrand for energy loss
def rad_energy_integrand(T, L, E, u_par=0.0, hard_pid=21, model='GLV'):
    """
    Compute the energy radiated in gluon spectrum per unit length of this step.

    Params:
        T : Temperature of plasma seen in this step
        L : The total pathlength traveled in plasma of this particle. Note that this should be total pathlength minus pathlength in fs.
        u_par : The parallel flow velocity of the medium
        hard_pid : The pdg hard parton id of the particle
        model : The energy loss model to use. Options are 'GLV' and 'BBMG'
    """
    FmGeV = 1/0.19732687

    # Average medium parameters
    mu = mu_DeBye(T)
    inv_lambda_val = inv_lambda(T, hard_pid=hard_pid, soft_pid=None)

    # Select radiation energy model and return appropriate energy per unit pathlength
    if model == 'BBMG':
        # Note that we apply FERMI GeV twice... Once for the t factor, once for the (int dt).
        return (config.jet.K_BBMG * ((FmGeV) ** 2) * L * (T ** 3)
                * zeta(q=-1) * (1 / np.sqrt(1 - (u_par ** 2)))
                * (1))
    elif model == 'GLV':
        # https://inspirehep.net/literature/539404
        # Note that we apply FERMItoGeV twice... Once for the t factor, once for the (int dt).
        # Set C_R, "quadratic Casimir of the representation R of SU(3) for the parton"
        if hard_pid == 21:
            # For a gluon it's the adjoint representation C_A = N_c = 3
            CR = 3
        elif np.abs(hard_pid) < 7:
            # For a quark it's the fundamental representation C_F = 4/3 in QCD
            CR = 4/3
        else:
            # Ill-defined for other partons
            logging.error("Ill-defined Casimir factor for radiative energy transfer!")
            CR = 0

        # Set alpha_s
        alphas = (config.constants.G**2) / (4*np.pi)

        # Calculate and return energy radiated per unit length of this step.
        return (CR * alphas / 2) * (((FmGeV) ** 2)
                                         * L
                                         * (mu**2)
                                         * inv_lambda_val
                                         * np.log(E / mu))
    else:
        return 0


# Integrand for energy loss
# https://journals.aps.org/prd/pdf/10.1103/PhysRevD.44.R2625
def coll_energy_loss_integrand(T, tau, E, hard_pid=21):
    FmGeV = 1/0.19732687
    nf = 2  # Source?

    # Set C_R, "quadratic Casimir of the representation R of SU(3) for the parton"
    if hard_pid == 21:
        # For a gluon it's the adjoint representation C_A = N_c = 3
        CR = 3
    elif np.abs(hard_pid) < 7:
        # For a quark it's the fundamental representation C_F = 4/3 in QCD
        CR = 4 / 3
    else:
        # Ill-defined for other partons
        logging.error("Ill-defined Casimir factor for radiative energy transfer!")
        CR = 0

    # Set alpha_s
    ALPHAS = (config.constants.G**2) / (4*np.pi)

    # Calculate and return energy loss per unit length of this step.
    mg = (config.constants.G * T / np.sqrt(3)) * np.sqrt(1 + (nf / 6))  # Thermal gluon mass, see paper
    return (-1) * FmGeV * CR * (3 / 4) * (8 * np.pi * (ALPHAS ** 2) / 3) * (1 + (nf / 6)) * (T ** 2) * np.log(
        (2 ** (nf / (2 * (6 + nf)))) * 0.920 * (np.sqrt(E * T) / mg))


# Collisional interaction momentum transfer public API
def collisional_delta(particle: hard_particles.Particle, medium: plasma.plasma_event, dtau: float) -> hard_particles.ParticleDelta:
    # Start counters
    dpx = 0
    dpy = 0
    dpz = 0

    # Gather particle and medium properties.
    p = particle.p3
    point = particle.coords
    temp = medium.temp(point)[0]
    if temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()
    u = np.array([float(medium.x_vel(point)[0]), float(medium.y_vel(point)[0]), float(medium.z_vel(point)[0])])

    # Get perp and parallel medium flow velocity
    uperp = utilities.perp_vec(a=u, b=p)
    upar = utilities.par_vec(a=u, b=p)

    # Get pathlength traveled
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    pathlength = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # Compute flow-induced broadening, add to momentum transfer.
    drift = drift_integrand(T=temp, u_perp=np.linalg.norm(uperp), u_par=np.linalg.norm(upar),
                                               E=particle.E, hard_pid=particle.id) * pathlength
    uperp_hat = uperp / np.linalg.norm(uperp)  # Unit vector in direction of u_perp
    drift_vec = drift * uperp_hat

    dpx += float(drift_vec[0])
    dpy += float(drift_vec[1])
    dpz += float(drift_vec[2])

    # Compute longitudinal momentum transfer
    dpx += 0.0
    dpy += 0.0
    dpz += 0.0

    return hard_particles.ParticleDelta(dpx=dpx, dpy=dpy, dpz=dpz)


# Collisional interaction momentum transfer public API using density gradient
def collisional_delta_linear_gradients(particle: hard_particles.Particle, medium: plasma.plasma_event, dtau: float) -> hard_particles.ParticleDelta:
    # Start counters
    dpx = 0
    dpy = 0
    dpz = 0

    # Gather particle and medium properties.
    p = particle.p3
    point = particle.coords
    temp = medium.temp(point)[0]
    if temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()
    u = np.array([float(medium.x_vel(point)[0]), float(medium.y_vel(point)[0]), float(medium.z_vel(point)[0])])
    gradtemp_vec = np.array([float(medium.temp_grad_x(point)[0]), float(medium.temp_grad_y(point)[0]), float(medium.temp_grad_z(point)[0])])

    # Get perp and parallel medium flow velocity
    uperp = utilities.perp_vec(a=u, b=p)
    upar = utilities.par_vec(a=u, b=p)

    # Get perp temperature gradient
    gradtempperp = utilities.perp_vec(a=gradtemp_vec, b=p)

    # Get pathlength traveled
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    pathlength = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # Compute flow-induced broadening, add to momentum transfer.
    drift = drift_integrand_linear_gradients(T=temp, u_perp=uperp, u_par=np.linalg.norm(upar),
                                             gradtemp=gradtempperp, tau=pathlength,
                                               E=particle.E, hard_pid=particle.id) * pathlength
    uperp_hat = uperp / np.linalg.norm(uperp)  # Unit vector in direction of u_perp
    drift_vec = drift * uperp_hat

    dpx += float(drift_vec[0])
    dpy += float(drift_vec[1])
    dpz += float(drift_vec[2])

    # Compute longitudinal momentum transfer
    dpx += 0.0
    dpy += 0.0
    dpz += 0.0

    return hard_particles.ParticleDelta(dpx=dpx, dpy=dpy, dpz=dpz)


# Radiative interaction momentum transfer public API
def rad_delta(particle: hard_particles.Particle, medium: plasma.plasma_event, dtau: float) -> hard_particles.ParticleDelta:
    # Start counters
    dpx = 0
    dpy = 0
    dpz = 0

    # Gather particle and medium properties.
    p = particle.p3
    point = particle.coords
    temp = medium.temp(point)[0]
    if temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()

    # Get total pathlength traveled in the plasma
    pathlength = particle.pathlength_since(medium.t0)

    # Get pathlength traveled in this step
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    delta_pathlength = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

    # Compute radiative longitudinal energy loss -- Note particle LOSES the energy radiated, so we have (-1) factor
    E_change = float((-1) * rad_energy_integrand(temp, pathlength, particle.E, hard_pid=particle.id,
                                      model='GLV') * delta_pathlength)

    # Add energy change to momentum parallel to particle motion (unit vector in direction of p)
    # Radiated energy does only comes from momentum of particle, not from mass.
    p_hat = p / np.linalg.norm(p)
    dpx += float(E_change * p_hat[0])
    dpy += float(E_change * p_hat[1])
    dpz += float(E_change * p_hat[2])


    return hard_particles.ParticleDelta(dpx=dpx, dpy=dpy, dpz=dpz)