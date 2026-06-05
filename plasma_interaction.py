import numpy as np
import config
import hard_particles
import plasma
import utilities
import logging
import os
import sys
from pathlib import Path
from scipy.ndimage import map_coordinates

from utilities import zeta, perp_vec

# Get the path of this file and import the radiation NN path
script_dir = str(Path(__file__).resolve().parent)
sys.path.append(os.path.join(script_dir, 'flow-rad-nn/'))
from train_radiation_nn import RadiationEmulatorInference

class HadronGas(Exception):
    """
    Raise to end evolution while computing an interaction if there is no event data for the coordinates of the particle
    """


class NoMedium(Exception):
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
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()
    u = np.array([float(medium.x_vel(point)[0]), float(medium.y_vel(point)[0]), float(medium.z_vel(point))])

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
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()
    u = np.array([float(medium.x_vel(point)[0]), float(medium.y_vel(point)[0]), float(medium.z_vel(point))])
    gradtemp_vec = np.array([float(medium.temp_grad_x(point)[0]), float(medium.temp_grad_y(point)[0]), float(medium.temp_grad_z(point))])

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
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
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


# Radiative interaction momentum transfer public using anisotropic model NN API
def aniso_rad_delta(particle: hard_particles.Particle, medium: plasma.plasma_event, k: np.ndarray) -> hard_particles.ParticleDelta:
    # Start counters
    dpx = 0
    dpy = 0
    dpz = 0

    # Align momentum transfer to coordinate system
    lf_momentum = lf_emission_momentum(k=k, particle=particle, medium=medium)
    dp = (-1) * lf_momentum

    dpx += float(dp[0])
    dpy += float(dp[1])
    dpz += float(dp[2])

    # Return particle delta
    return hard_particles.ParticleDelta(dpx=dpx, dpy=dpy, dpz=dpz)


# Radiation distribution summoner
def aniso_rad_dist(particle: hard_particles.Particle, medium: plasma.plasma_event,
                   kz_values: np.ndarray, kx_values: np.ndarray, ky_values: np.ndarray, dtau: float, nn=None):
    """
    Function that generates a 3D numpy array of the number distribution of emitted gluons over the current step in the
    medium. ky_values should be an even number of points symmetric about 0 so we can mirror points along this axis.

    Returns in the parton frame organized (kx, ky, kz)
    """
    assert len(ky_values) % 2 == 0  # Array has an even number of entries
    # assert np.array_equal(ky_values, -ky_values[::-1])  # Array is symmetric about 0
    hbar = 0.1973269804  # GeV * fm

    # Gather particle and medium properties.
    if particle.isq:
        CR = 4/3
    elif particle.isg:
        CR = 3
    else:
        # Default to quark CF
        CR = 4/3
    p = particle.p3
    point = particle.coords
    temp = medium.temp(point)[0]
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()
    u = np.array([float(medium.x_vel(point)[0]), float(medium.y_vel(point)[0]), float(medium.z_vel(point))])
    uperp = utilities.perp_vec(a=u, b=p)

    # Get pathlength traveled in this step
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    delta_pathlength = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

    # Warn if we're outside our training domain
    # if particle.tau + delta_pathlength / hbar > np.amax(nn.X[5]):
    #     logging.warning("Particle pathlength is outside of training domain! Good luck!")
    # if temp > np.amax(nn.X[7]) or temp < np.amin(nn.X[7]):
    #     logging.warning("Temperature is outside of training domain! Good luck!")
    # if np.linalg.norm(uperp) > np.amax(nn.X[6]) or np.linalg.norm(uperp) < np.amin(nn.X[6]):
    #     logging.warning("Perp. velocity is outside of training domain! Good luck!")
    if particle.tau + delta_pathlength / hbar > 50.0:
        logging.warning("Particle pathlength is outside of training domain! Good luck!")
    if temp > 0.650 or temp < 0.150:
        logging.warning("Temperature is outside of training domain! Good luck!")
    if np.linalg.norm(uperp) > 0.9 or np.linalg.norm(uperp) < 0.0:
        logging.warning("Perp. velocity is outside of training domain! Good luck!")

    # Compute number distribution of radiation generated in this step
    dtau_rad_dist = nn.compute_dNd3k_grid(
        E=particle.E0,  # Use E0 to avoid rescaling the meaning of x between steps
        z0=particle.tau / hbar,  # tau is in fm, need to give to NN in GeV^{-1}
        zf=(particle.tau + delta_pathlength) / hbar,  # tau & dtau are in fm, need to give to NN in GeV^{-1}
        u_perp=np.linalg.norm(uperp),
        T=temp,
        g=config.constants.G,
        kx_values=kx_values,
        ky_values=ky_values[len(ky_values) // 2 ::],  # Compute only for positive ky values
        kz_values=kz_values[len(kz_values) // 2 ::]  )  # Compute only for positive kz values

    # Mirror across ky -- Flip array, then concat along that axis.
    dtau_rad_dist = np.concat((np.flip(dtau_rad_dist, axis=1), dtau_rad_dist), axis=1)

    # Fill zeroes for negative kz values and concat along that axis
    dtau_rad_dist = np.concat((np.zeros_like(dtau_rad_dist), dtau_rad_dist), axis=2)

    return CR * dtau_rad_dist


def lf_emission_momentum(k: np.ndarray, particle, medium: plasma.plasma_event):
    """
    Function to transform the 3-momentum of an emission from the jet frame to the lab frame
    """

    # Gather particle and medium properties.
    p = particle.p3
    point = particle.coords
    u = np.array([float(medium.x_vel(point)[0]), float(medium.y_vel(point)[0]), float(medium.z_vel(point))])
    uperp = perp_vec(a=u, b=p)

    # By construction, the gluon kinematics correspond to:
    k_z_hat = p / np.linalg.norm(p)  # Direction of k_z is parallel to the hard particle
    k_x_hat = uperp / np.linalg.norm(uperp)  # Direction of k_x is parallel to the transverse flow
    k_y_hat = np.cross(k_z_hat, k_x_hat)  # Direction of k_y is perp to both of the above, k_x x k_y = k_z, permute to k_z x k_x = k_y

    # Return transformed momentum 3-vector
    return np.array(k[0]*k_x_hat + k[1]*k_y_hat + k[2]*k_z_hat)


def rotate_rad_dist(particle: hard_particles.Particle, medium: plasma.plasma_event, rad_dist: np.ndarray,
                    kx_values: np.ndarray, ky_values: np.ndarray, kz_values: np.ndarray):
    """
    Function that rotates a radiation distribution in kx, ky, kz in the parton frame into the lab frame.

    """
    # Find axes
    # Gather particle and medium properties.
    p = particle.p3
    point = particle.coords
    u = np.array([float(medium.x_vel(point)[0]), float(medium.y_vel(point)[0]), float(medium.z_vel(point))])
    uperp = perp_vec(a=u, b=p)

    # By construction, the gluon kinematics correspond to:
    k_z_hat = p / np.linalg.norm(p)  # Direction of k_z is parallel to the hard particle
    k_x_hat = uperp / np.linalg.norm(uperp)  # Direction of k_x is parallel to the transverse flow
    k_y_hat = np.cross(k_z_hat, k_x_hat)  # Direction of k_y is perp to both of the above, k_x x k_y = k_z, permute to k_z x k_x = k_y

    # # Transpose rad_dist from (kz, kx, ky) -> (kx, ky, kz)
    # dist_kxkykz = rad_dist.transpose(1, 2, 0)  # Shape: (m, m, n)
    dist_kxkykz = rad_dist  # Should already be in (kx, ky, kz), shape (m, m, n)

    # Compute the input integral (np.trapezoid handles integration of arbitrary spacing via coordinates)
    integral_before = np.trapezoid(
        np.trapezoid(
            np.trapezoid(dist_kxkykz, kz_values, axis=2),
            ky_values, axis=1),
        kx_values, axis=0)

    # Build rotation matrix R: columns are k_x_hat, k_y_hat, k_z_hat in (x,y,z) space.
    # R transforms a (kx, ky, kz) vector to (x, y, z): v_xyz = R @ v_k
    # R^T (= R^-1 for orthonormal R) transforms (x, y, z) back to (kx, ky, kz): v_k = R^T @ v_xyz
    R = np.column_stack([k_x_hat, k_y_hat, k_z_hat])  # Shape: (3, 3)

    # Build a meshgrid of output (x, y, z) coordinates using the same grid ranges as the input
    x_values = kx_values
    y_values = ky_values
    z_values = kz_values

    # Build output grid in (x, y, z) and find corresponding (kx, ky, kz) source coordinates via R^T
    xg, yg, zg = np.meshgrid(x_values, y_values, z_values, indexing='ij')  # Each shape: (m, m, n)
    xyz = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=0)  # Shape: (3, m*m*n)

    # Apply inverse rotation to map output (x,y,z) grid back to source (kx,ky,kz) coordinates
    k_coords = R.T @ xyz  # Shape: (3, m*m*n); rows are kx, ky, kz source coords

    # Convert source (kx, ky, kz) coordinates to fractional array indices in dist_kxkykz
    # Interpolates between coordinates to allow for any grid spacing in any axis (log, linear, etc.)
    # Convert source (kx, ky, kz) coordinates to fractional array indices in dist_kxkykz
    # Use searchsorted-based linear mapping instead of np.interp to allow out-of-bounds indices
    # (np.interp clamps to edge values, preventing map_coordinates from correctly zeroing them)
    def coords_to_index(coords, grid):
        """Linearly map coordinate values to fractional array indices, allowing out-of-bounds."""
        # For uniform or non-uniform grids: find fractional index by linear interpolation of the inverse map
        indices = np.interp(coords, grid, np.arange(len(grid)),
                            left=-(len(grid)), right=2 * len(grid))  # force OOB indices far outside
        return indices

    idx_kx = coords_to_index(k_coords[0], kx_values)
    idx_ky = coords_to_index(k_coords[1], ky_values)
    idx_kz = coords_to_index(k_coords[2], kz_values)

    # idx_kx = np.interp(k_coords[0], kx_values, np.arange(len(kx_values)))
    # idx_ky = np.interp(k_coords[1], ky_values, np.arange(len(ky_values)))
    # idx_kz = np.interp(k_coords[2], kz_values, np.arange(len(kz_values)))

    # Apply rotation
    # Interpolate dist_kxkykz at the source fractional indices; points outside bounds are set to 0
    # Use scipy.ndimage.map_coordinates to perform the interpolation of points into the new axes.
    rotated_flat = map_coordinates(dist_kxkykz, [idx_kx, idx_ky, idx_kz], order=1, mode='constant', cval=0.0)
    rotated_rad_dist = rotated_flat.reshape(len(x_values), len(y_values), len(z_values))  # Shape: (m, m, n)

    # Rescale to conserve the numerical integral.
    # Any signal whose rotated source coordinates fell outside the input grid was set to 0 by
    # map_coordinates. We compensate by rescaling the captured shape to match the original integral.
    # (np.trapezoid handles integration of arbitrary spacing via coordinates)
    integral_after = np.trapezoid(
        np.trapezoid(
            np.trapezoid(rotated_rad_dist, z_values, axis=2),
            y_values, axis=1),
        x_values, axis=0)

    if integral_after != 0.0:
        rotated_rad_dist *= integral_before / integral_after

    change = (integral_after - integral_before) / integral_before
    if change > 0.1:
        logging.warning(f"Radiation distribution rotation change: {change*100}%")

    return rotated_rad_dist