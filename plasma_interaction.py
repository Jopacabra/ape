import numpy as np
from xarray.ufuncs import invert

import config
import hard_particles
import plasma
import utilities
import logging
import os
import sys
from pathlib import Path
from scipy.ndimage import map_coordinates

from utilities import zeta, perp_vec, rng

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
def inv_lambda_rhograd(T, gradtemp, hard_pid=21, soft_pid=None):
    """
    By default:
    We apply a reciprocal summation between the cross-section times density for medium gluons and for medium quarks
    to get the mean free path as in https://inspirehep.net/literature/1725162
    """

    if soft_pid is None:
        # Assumes two light quark flavors in the medium, including their antiparticles
        return (sigma(temp=T, hard_pid=hard_pid, soft_pid=21) * gradrho(temp=T, gradtemp=gradtemp, soft_pid=21)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=1) * gradrho(temp=T, gradtemp=gradtemp, soft_pid=1)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=-1) * gradrho(temp=T, gradtemp=gradtemp, soft_pid=-1)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=2) * gradrho(temp=T, gradtemp=gradtemp, soft_pid=2)
                + sigma(temp=T, hard_pid=hard_pid, soft_pid=-2) * gradrho(temp=T, gradtemp=gradtemp, soft_pid=-2))
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


# Integrand for mean drift including linear gradients of transverse flow
def drift_integrand_linear_gradients(T, u_perp, u_par, p, grad_u, grad_T, z_position, E, x_perp=np.array([0,0,0]), hard_pid=21):
    """
    Note that as of now we're including ONLY linear gradients of density in the presence of flow. We neglect gradients
    of the debye mass and gradients of the flow.
    """
    HBARC = 0.1973269804  # GeV * fm

    # Compute vector (u_perp)_i  (grad u_perp)_{ij}
    ugradu = np.einsum('i,ij->j', u_perp, grad_u)
    xperpgradu = np.einsum('i,ij->j', x_perp, grad_u)

    # Compute matrix determinant -- phase space contraction
    p_hat = p / np.linalg.norm(p)
    e1, e2 = utilities.transverse_basis(p_hat)
    detM = utilities.det_M(np.array([grad_u]), np.array([u_par]), np.array([z_position]), e1, e2).item()  # get float from shape (1,) array
    invdetM = 1 / detM
    # logging.debug(f"1/detM = {invdetM}")

    # Get temperature gradient magnitude and direction
    grad_T_mag = np.linalg.norm(grad_T)
    if grad_T_mag > 0.0:
        grad_T_hat = grad_T / grad_T_mag
    else:
        grad_T_hat = np.array([0, 0, 0])

    # Compute inverse mfp and debye mass
    inv_lambda_val = inv_lambda(T, hard_pid=hard_pid, soft_pid=None)
    grad_rho_inv_lambda = inv_lambda_rhograd(T, grad_T_mag)
    mu = mu_DeBye(T)

    # Returns a 3-vector of the broadening per unit length in this step.
    grad_uperp = ((1/HBARC) * (3 / E)
            * invdetM  # Phase space modification from gradients of transverse flow
            * (mu **2)
            * inv_lambda_val  # Inverse MFP
            * np.log(E / mu)
            * (1 / (1 - u_par))
            * (u_perp - xperpgradu - (ugradu / (1 - u_par))*z_position)  # Vector difference!
            )
    # print(grad_uperp)

    grad_rho = ((1/HBARC) * (3 / E)
                * (mu ** 2)
                * grad_rho_inv_lambda  # Inverse MFP ONLY from gradrho
                * np.log(E / mu)
                * (1 / (1 - u_par))
                * np.dot((- xperpgradu - (u_perp / (1 - u_par))*z_position), grad_T_hat)
                ) * grad_T_hat
    # print(grad_rho)

    return config.jet.K_F_DRIFT * (grad_uperp)# + grad_rho)



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
def collisional_delta(particle: hard_particles.Particle, medium: plasma.plasma, dtau: float) -> hard_particles.ParticleDelta:
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
    upar_scalar = np.sign(np.dot(upar, p)) * np.linalg.norm(upar)

    # Get pathlength traveled
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    pathlength = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # Compute flow-induced broadening, add to momentum transfer.
    drift = drift_integrand(T=temp, u_perp=np.linalg.norm(uperp), u_par=upar_scalar,
                                               E=particle.E, hard_pid=particle.id) * pathlength
    uperp_mag = np.linalg.norm(uperp)
    if uperp_mag > 0.0:
        uperp_hat = uperp / uperp_mag  # Unit vector in direction of u_perp
    else:
        uperp_hat = np.array([0, 0, 0])
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
def collisional_delta_linear_gradients(particle: hard_particles.Particle, medium: plasma.plasma, dtau: float,
                                       scheme="MC_approx") -> hard_particles.ParticleDelta:
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
    upar_scalar = np.sign(np.dot(upar, p)) * np.linalg.norm(upar)

    # Get grad_u_perp matrix
    grad_x_u_x = medium.grad_x_u_x(point).item()
    grad_x_u_y = medium.grad_x_u_y(point).item()
    grad_x_u_z = 0  # plasma_object.grad_x_u_z(coords)

    grad_y_u_x = medium.grad_y_u_x(point).item()
    grad_y_u_y = medium.grad_y_u_y(point).item()
    grad_y_u_z = grad_x_u_z  # plasma_object.grad_y_u_z(coords)

    grad_z_u_x = 0  # plasma_object.grad_z_u_x(coords)
    grad_z_u_y = grad_z_u_x  # plasma_object.grad_z_u_y(coords)
    grad_z_u_z = grad_z_u_x  # plasma_object.grad_z_u_z(coords)
    grad_u = np.array([[grad_x_u_x, grad_x_u_y, grad_x_u_z],  # row i=x
        [grad_y_u_x, grad_y_u_y, grad_y_u_z],  # row i=y
        [grad_z_u_x, grad_z_u_y, grad_z_u_z],  # row i=z
    ])  # shape (N, 3, 3)

    # Get grad_T
    grad_T = np.array([medium.temp_grad_x(point).item(), medium.temp_grad_y(point).item(), medium.temp_grad_z(point).item()])

    # Get pathlength traveled
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    pathlength = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    total_pathlength = particle.pathlength_since(particle.tau_0) + (pathlength / 2)

    # Determine expansion point x_{\perp 0}
    if scheme == "MC_approx":
        """
        Our particle's recorded position relative to a straight line traj is the "average position", up to 
        fluctuations in drift. We will take the approximation of accumulated (p_perp / E)z
        """
        x_perp = total_pathlength * (utilities.perp_vec(particle.p3, particle.p30))/particle.E0  # 3-vector!
    elif scheme == "theory":
        x_perp = np.array([0,0,0])  # 3-vector!

    # Compute flow-induced broadening (integrand times step pathlength), add to momentum transfer.
    drift_vec = drift_integrand_linear_gradients(T=temp, u_perp=uperp, u_par=upar_scalar, p=particle.p3,
                                                 grad_u=grad_u, grad_T=grad_T, z_position=total_pathlength,
                                                 E=particle.E, hard_pid=particle.id, x_perp=x_perp) * pathlength

    dpx += float(drift_vec[0])
    dpy += float(drift_vec[1])
    dpz += float(drift_vec[2])

    # Compute longitudinal momentum transfer
    dpx += 0.0
    dpy += 0.0
    dpz += 0.0

    return hard_particles.ParticleDelta(dpx=dpx, dpy=dpy, dpz=dpz)


# Radiative interaction momentum transfer public API
def rad_delta(particle: hard_particles.Particle, medium: plasma.plasma, dtau: float) -> hard_particles.ParticleDelta:
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
def aniso_rad_delta(particle: hard_particles.Particle, medium: plasma.plasma, k: np.ndarray) -> hard_particles.ParticleDelta:
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
def aniso_rad_dist(particle: hard_particles.Particle, medium: plasma.plasma,
                   kz_values: np.ndarray, kx_values: np.ndarray, ky_values: np.ndarray, dtau: float, nn=None):
    """
    Function that generates a 3D numpy array of the number distribution of emitted gluons over the current step in the
    medium. ky_values should be an even number of points symmetric about 0 so we can mirror points along this axis.

    Returns in the parton frame organized (kx, ky, kz)
    """
    # assert len(ky_values) % 2 == 0  # Array has an even number of entries
    assert np.array_equal(ky_values, -ky_values[::-1])  # Array is symmetric about 0
    assert np.all(kz_values > 0), "kz_values must contain only positive numbers"
    hbarc = 0.1973269804  # GeV * fm

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
    uperp = np.linalg.norm(utilities.perp_vec(a=u, b=p))

    # Get pathlength traveled in this step
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    delta_pathlength = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

    # Perform longitudinal boost to longitudinal rest frame of the fluid -- ???
    """
    Right now, we're only incorporating the transverse flow. The calculation was done in the longitudinal rest frame
    of the fluid. We can easily get around this by performing a boost to the longitudinal rest frame of the fluid, then
    simply boosting the resulting distribution back to the lab frame. In truth, there may be gauge invariance problems
    that are related to this assumption.
    
    For this testing version, we have not yet implemented the longitudinal boost.
    """

    # Warn if we're outside our training domain
    if particle.tau + delta_pathlength / hbarc > 50.0:
        logging.warning("Particle pathlength is outside of training domain! Good luck!")
    if temp > 0.650 or temp < 0.150:
        logging.warning("Temperature is outside of training domain! Good luck!")
    if uperp > 0.9 or uperp < 0.0:
        logging.warning("Perp. velocity is outside of training domain! Good luck!")

    # Compute number distribution of radiation generated in this step
    dtau_rad_dist = nn.compute_dNd3k_grid(
        E=particle.E,
        z0=particle.tau / hbarc,  # tau is in fm, need to give to NN in GeV^{-1}
        # zf=(particle.tau + delta_pathlength) / hbarc,  # tau & dtau are in fm, need to give to NN in GeV^{-1}
        u_perp=uperp,
        T=temp,
        g=config.constants.G,
        kx_values=kx_values,
        ky_values=ky_values[ky_values >= 0],  # Compute only for ky >= 0
        kz_values=kz_values)  # Should pass only positive kz values

    # Mirror across ky -- Flip array excluding zero element, then concat along that axis.
    dtau_rad_dist = np.concat((np.flip(dtau_rad_dist[:, 1:, :], axis=1), dtau_rad_dist), axis=1)

    # Perform longitudinal boost to back to lab frame -- ???
    """
    For this testing version, we have not yet implemented the longitudinal boost.
    """

    return CR * dtau_rad_dist


def lf_emission_momentum(k: np.ndarray, particle, medium: plasma.plasma):
    """
    Function to transform the 3-momentum of an emission from the jet frame to the lab frame
    """

    # Gather particle and medium properties.
    p = particle.p3
    point = particle.coords
    u = np.array([float(medium.x_vel(point)[0]), float(medium.y_vel(point)[0]), float(medium.z_vel(point))])
    uperp = perp_vec(a=u, b=p)
    uperp_mag = np.linalg.norm(uperp)

    # By construction, the gluon kinematics correspond to:
    k_z_hat = p / np.linalg.norm(p)  # Direction of k_z is parallel to the hard particle
    if uperp_mag > 0.0:
        k_x_hat = uperp / uperp_mag  # Direction of k_x is parallel to the transverse flow
        k_y_hat = np.cross(k_z_hat, k_x_hat)  # Direction of k_y is perp to both of the above, k_x x k_y = k_z, permute to k_z x k_x = k_y
    else:
        # Direction of transverse flow uncertain... Orthonormal basis perp to k_z_hat, spectrum should be symmetric.
        logging.warning("Exactly zero uperp. Radiation using default orthonormal basis.")
        k_x_hat, k_y_hat = utilities.transverse_basis(k_z_hat)

    # Return transformed momentum 3-vector
    return np.array(k[0]*k_x_hat + k[1]*k_y_hat + k[2]*k_z_hat)


def rotate_rad_dist(particle: hard_particles.Particle, medium: plasma.plasma, rad_dist: np.ndarray,
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


def E_gluons(particle: hard_particles.Particle, medium: plasma.plasma, dtau: float):
    """
    This function returns the analytically computed energy of emitted gluons at first order in opacity with infinite
    kinematic bounds, as in https://arxiv.org/pdf/nucl-th/0012092 Eq. 9, for a single step.
    """
    # Gather particle and medium properties.
    HBARC = 0.197327  # GeV·fm
    ALPHAS = (config.constants.G**2) / (4*np.pi)
    if particle.isq:
        CR = 4 / 3
    elif particle.isg:
        CR = 3
    else:
        # Default to quark CF
        CR = 4 / 3
    E = particle.E
    point = particle.coords
    temp = medium.temp(point)[0]
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()

    # Get pathlength traveled in this step
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    delta_L = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

    # The (z-z0) factor should be identified with the current pathlength in the plasma, particle.tau. This retains the
    # overall L^2 behavior of the function. The $\int dz$ factor gives us the pathlength in this step.
    intdz = delta_L / HBARC
    L0 = (max(particle.tau - medium.t0, 0) / HBARC)  # Pathlength already traversed in the plasma at beginning of step
    L = (L0 + L0 + intdz) / 2  # Average pathlength in plasma of step, preventing 0.
    mu = mu_DeBye(T=temp)

    return ((2 * CR * ALPHAS/np.pi) * intdz * (mu ** 2) * inv_lambda(T=temp, hard_pid=particle.id)
            * L * np.log(E / mu))


def N_gluons(particle: hard_particles.Particle, medium: plasma.plasma, dtau: float):
    """
    This function returns the analytically computed number of emitted gluons at first order in opacity with infinite
    kinematic bounds, as in https://arxiv.org/pdf/nucl-th/0012092 Eq. 7

    We take the opposite limit as they do to arrive at Eq. 9, assuming small $x << x_c = (L \mu^2 / (2 E))$.
    Then, we can apply a factor of (1/(xE)) and integrate over x from $x_{min} = \mu/E$ to $x_{max} = x_c$. We need
    a minimum value of x here because, while $dI/dx \propto \log(1/x)$ is integrable to zero,
    $dN/dx \propto \log(1/x)/x$ is not.
    """
    # Gather particle and medium properties.
    HBARC = 0.197327  # GeV·fm
    ALPHAS = (config.constants.G**2) / (4*np.pi)
    if particle.isq:
        CR = 4 / 3
    elif particle.isg:
        CR = 3
    else:
        # Default to quark CF
        CR = 4 / 3
    E = particle.E
    point = particle.coords
    temp = medium.temp(point)[0]
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()

    # Get pathlength traveled in this step
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    delta_L = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

    # The (z-z0) factor should be identified with the current pathlength in the plasma, particle.tau. This retains the
    # overall L^2 behavior of the function. The $\int dz$ factor gives us the pathlength in this step.
    intdz = delta_L / HBARC
    L0 = (max(particle.tau - medium.t0, 0) / HBARC)
    L = (L0 + L0 + intdz) / 2  # Average pathlength in plasma of step, preventing 0.
    mu = mu_DeBye(T=temp)
    xmin = mu / E
    inv_lambda_val = inv_lambda(T=temp, hard_pid=particle.id)
    log_factor = np.log(L * (mu ** 2) / (2 * E * xmin))

    # Note here that the 1/2 inside the log is scheme dependent. It also is a constant, so leading log approximations
    # often drop it. Don't be too perturbed by its presence or absence in different works.
    return ((CR * ALPHAS/np.pi) * intdz * inv_lambda_val
            * log_factor**2)


def sample_rad_dist(rad_dist, kx_values, ky_values, kz_values, N_samples=1, kin_cut=True, mu=0.3, E=10.0):
    """
    Function to sample radiation distribution for kx, ky, kz.
    Treat dI/(dxdkxdky) as an (unnormalized) 3D probability density and draw N_samples points (kx, ky, kz) from it.
    """
    # Build a normalized flat PDF, then a CDF
    # Compute bin widths using gradient (handles non-uniform spacing)
    dkx = np.gradient(kx_values)  # shape: (len(kx_values),)
    dky = np.gradient(ky_values)  # shape: (len(ky_values),)
    dkz = np.gradient(kz_values)  # shape: (len(kz_values),)

    # Build 3D volume element array via outer products
    dV = dkx[:, None, None] * dky[None, :, None] * dkz[None, None, :]  # shape: (Nkx, Nky, Nkz)

    # Flatten the 3D intensity array into a 1D array of probabilities (note multiplication by bin volume)
    I_flat = (rad_dist * dV).ravel()
    I_flat_pos = np.clip(I_flat, 0, None)  # ensure non-negative  # ensure non-negative
    pdf = I_flat_pos / I_flat_pos.sum()  # normalize to sum to 1
    cdf = np.cumsum(pdf)  # build CDF
    cdf[-1] = 1.0  # Force exact upper bound — removes all floating point slop

    failed = False
    for i in np.arange(0, 1000):  # Limit to 1000 attempts, so we can't have any soft locking.
        # Draw uniform samples and find where they land in the CDF
        uniform_samples = 1.0 - rng.uniform(size=N_samples)
        flat_indices = np.searchsorted(cdf, uniform_samples, side="right")  # shape: (N_samples,)

        # Convert flat indices back to 3D grid indices
        ikx, iky, ikz = np.unravel_index(flat_indices, rad_dist.shape)

        # Look up the corresponding coordinate values and add jitter about the bin, so we don't sample exactly on the points
        sampled_kx = kx_values[ikx] + rng.uniform(-0.5, 0.5, size=N_samples) * dkx[ikx]
        sampled_ky = ky_values[iky] + rng.uniform(-0.5, 0.5, size=N_samples) * dky[iky]
        sampled_kz = kz_values[ikz] + rng.uniform(-0.5, 0.5, size=N_samples) * dkz[ikz]

        if kin_cut:
            failed = False
            # Check if the sampled point is within the kinematic cuts
            for kx, ky, kz in zip(sampled_kx, sampled_ky, sampled_kz):
                kperp2 = (kx**2 + ky**2)
                x = kz / E
                if ((kperp2 >= mu**2) and (kperp2 < (4*(E**2) * min(x**2, (1-x)**2) - mu**2)) and (x > mu/(2*E))
                        and (x < 1 - mu/(2*E))):
                    pass  # Success
                else:
                    failed = True
                    break
            if failed:
                continue  # Sample another set.
            break  # Successful cut!


        else:
            break

    if failed:
        return None

    # Stack values
    emission_momentum = np.column_stack([sampled_kx, sampled_ky, sampled_kz])

    if N_samples == 1:
        return np.reshape(emission_momentum, 3)
    else:
        return emission_momentum


def aniso_nn(particle: hard_particles.Particle, plasma_object: plasma.plasma, dtau: float, nn):
    """
    Use a Neural Network emulator to compute the radiation spectrum for this particle in this macrostep.

    Perform a simplified Ogatta thinning, assuming a constant rate between emissions (within this step).

    Then, sample the kinematics of the emitted particles from the distribution.
    """
    # Hard coded options
    fix_rate = True  # Whether or not to replace the integral of the radiation spectrum with the analytic estimate.

    # Compute radiation kinematic bounds -- see https://arxiv.org/abs/nucl-th/0112071
    point = particle.coords
    temp = plasma_object.temp(point).item()
    mu = mu_DeBye(temp)
    x_min = mu / (2 * particle.E)
    x_max = 1 - x_min

    # Create bins of kz
    x_min_pow = np.log10(x_min)  # minimum power of 10 in x to compute
    x_max_pow = np.log10(x_max)  # maximum power of 10 in x to compute
    num_k_points = 20  # number of log-spaced points in kz to compute
    x_values = np.logspace(x_min_pow, x_max_pow, num_k_points // 2)
    k_z_pos_values = x_values * particle.E
    k_z_values = k_z_pos_values  # no need for negative kz now!

    # Create bins in k_perp
    # Note: maximum of ((Min[x^2, x(1-x)] * 4 * E_0^2) - mu^2) --> E_0^2 - mu^2
    # Added a factor of 0.25, because everything else is expensive numerically small probabilities
    # !!!!!!!!!!!!! Revisit this later !!!!!!!!!!!!!
    num_k_perp_points = 25  # num or (num - 1) of points in kx & ky to compute -- 0 added
    k_perp_pos_values = np.linspace(0, 0.25 * np.sqrt(particle.E ** 2 - mu ** 2),
                                    num_k_perp_points // 2)
    k_perp_values = np.concatenate((-np.flip(k_perp_pos_values[1:]), k_perp_pos_values))

    _, _, x_grid = np.meshgrid(k_perp_values, k_perp_values, x_values, indexing='ij')

    emission_momenta = []

    # Create a subdivision counter to march through this step, sampling the position of the next emission based on the
    # radiation rate. Exit when we either finish the step or the particle reaches our desired medium scale.
    tau = particle.tau
    total_kz = 0
    particle_p_norm = np.linalg.norm(particle.p3)
    while tau < tau + dtau and total_kz < particle_p_norm - config.jet.EMIN:
        # Compute radiation distribution from this macrostep -- returned in (kx, ky, kz) in parton frame
        dtau_rad_dist = aniso_rad_dist(particle=particle, medium=plasma_object, dtau=dtau,
                                       kx_values=k_perp_values,
                                       ky_values=k_perp_values,
                                       kz_values=k_z_values, nn=nn)

        # Compute integrals of complete radiation distribution
        # (np.trapezoid handles integration of arbitrary spacing via coordinates)
        total_number = np.trapezoid(
            np.trapezoid(
                np.trapezoid(dtau_rad_dist, k_z_values, axis=2),  # Integrate over kz -> shape: (n_kx, n_ky)
                k_perp_values, axis=1),  # Integrate over ky -> shape: (n_kx)
            k_perp_values, axis=0)  # Integrate over kx -> scalar

        # Approximate radiation rate
        if fix_rate:
            rate = N_gluons(particle=particle, medium=plasma_object, dtau=dtau) / dtau
        else:
            rate = total_number / dtau

        # Ogatta-thinning style sample for next emission position
        tau_to_emit = rng.exponential(1.0 / rate)
        next_tau = tau + tau_to_emit

        # If the next emission lies outside this macro timestep, break and stop emitting in this step
        if next_tau > tau + dtau:
            break

        # Otherwise, accept the emission & sample the distribution for emission kinematics in the radiation frame
        k = sample_rad_dist(dtau_rad_dist, N_samples=1,
                            kx_values=k_perp_values, ky_values=k_perp_values,
                            kz_values=k_z_values, mu=mu, E=particle.E)
        if k is None:
            logging.warning("Gluon kinematics rejected. Skipping emissions.")
            tau = next_tau
            continue
        logging.debug(f"Emitting gluon! Radiation frame info:")

        # If we rescale energies, do it!
        # Compute expected energy of emitted gluons
        logging.debug(f"p = {particle.p3} GeV")
        logging.debug(f"k = {k} GeV")

        # Check if gluon is backward facing -- This shouldn't happen from a single step's radiation,
        # but summing the distribution over multiple steps "turns" the coordinate system such that
        # it has a nonzero total probability
        if k[2] < 0.0:
            logging.warning(
                "!\n!\n!\nEmitted gluon is backward facing. Not good!\n!\n!\n!")

        # Transform emission momentum to lab frame
        k = lf_emission_momentum(k=k, particle=particle, medium=plasma_object)
        logging.debug(f"k = {k} GeV")

        # Append momenta to list for this step
        emission_momenta.append(k)
        total_kz += np.linalg.norm(k)

        # Step forward in time and reiterate
        tau = next_tau

    # Return all of the emissions for this macrostep
    return emission_momenta