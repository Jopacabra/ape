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
    temp = medium.temp(point).item()
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()
    u = np.array([float(medium.x_vel(point).item()), float(medium.y_vel(point).item()), float(medium.z_vel(point).item())])

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
    temp = medium.temp(point).item()
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()
    u = np.array([float(medium.x_vel(point).item()), float(medium.y_vel(point).item()), float(medium.z_vel(point).item())])

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
    temp = medium.temp(point).item()
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
def aniso_rad_dist(E: float, tau: float, temp: float, uperp: float,
                   x_values: np.ndarray, k_perp_values: np.ndarray, phi_values: np.ndarray,
                   dtau: float, nn=None):
    """
    Function that generates a 3D numpy array of the number distribution of emitted gluons over the current step in the
    medium, up to a casimir factor (Don't forget to apply it to the results of this function!).

    Returns in the parton frame organized (kperp, phi, x)
    """
    hbarc = 0.1973269804  # GeV * fm

    # Perform longitudinal boost to longitudinal rest frame of the fluid
    """
    Right now, we're only incorporating the transverse flow. The calculation was done in the longitudinal rest frame
    of the fluid. We can easily get around this by performing a boost to the longitudinal rest frame of the fluid, then
    simply boosting the resulting distribution back to the lab frame. In truth, there may be gauge invariance problems
    that are related to this assumption.
    
    For this testing version, we have not yet implemented the longitudinal boost.
    """

    # Warn if we're outside our training domain
    if (tau + dtau) / hbarc > 50.0:
        logging.warning("Particle pathlength is outside of training domain! Good luck!")
    if temp > 0.650 or temp < 0.150:
        logging.warning("Temperature is outside of training domain! Good luck!")
    if uperp > 0.9 or uperp < 0.0:
        logging.warning("Perp. velocity is outside of training domain! Good luck!")

    # Compute number distribution of radiation generated in this step
    dtau_rad_dist = nn.compute_dNdxd2k_grid(
        E=E,
        z0=tau / hbarc,  # tau is in fm, need to give to NN in GeV^{-1}
        u_perp=uperp,
        T=temp,
        g=config.constants.G,
        k_perp_values=k_perp_values,
        phi_values=phi_values,
        x_values=x_values,
        mu=mu_DeBye(T=temp))  # For kinematic cuts

    # Perform longitudinal boost to back to lab frame
    """
    For this testing version, we have not yet implemented the longitudinal boost.
    """

    return dtau_rad_dist


def lf_emission_momentum(k: np.ndarray, particle, medium: plasma.plasma):
    """
    Function to transform the 3-momentum of an emission from the jet frame to the lab frame
    """

    # Gather particle and medium properties.
    p = particle.p3
    point = particle.coords
    u = np.array([float(medium.x_vel(point).item()), float(medium.y_vel(point).item()), float(medium.z_vel(point).item())])
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
    temp = medium.temp(point).item()
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
    temp = medium.temp(point).item()
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


def sample_rad_dist(rad_dist, k_perp_values, phi_values, x_values, N_samples=1, kin_cut=True, mu=0.3, E=10.0):
    """
    Function to sample radiation distribution for kx, ky, kz.
    Treat dI/(dx d^2 k) as an (unnormalized) 3D probability density and draw N_samples points (k_perp, phi, x).
    Then, convert to cartesian coordinate vector (kx, ky, kz) and return.
    """
    # Compute kinematic bounds
    x_min = mu / E
    x_max = 1.0 - mu / E
    kperp_min = mu
    kperp_max = lambda x : np.sqrt((E ** 2) * np.minimum(x ** 2, (1.0 - x) ** 2) - mu ** 2)

    # Build a normalized flat PDF, then a CDF
    # Compute bin widths using gradient (handles non-uniform spacing)
    dkperp = np.gradient(k_perp_values)  # shape: (len(k_perp_values),)
    dphi = np.full_like(phi_values, 2 * np.pi / len(phi_values))  # shape: (len(phi_values),) -- phi periodic!
    dx = np.gradient(x_values)  # shape: (len(x_values),)

    # Build 3D volume element array via outer products -- includes Jacobian factor for the angular differential
    dV = (k_perp_values * dkperp)[:, None, None] * dphi[None, :, None] * dx[None, None, :]  # shape: (Nkperp, Nphi, Nx)

    # Flatten the 3D intensity array into a 1D array of probabilities (note multiplication by bin volume)
    I_flat = (rad_dist * dV).ravel()
    I_flat_pos = np.clip(I_flat, 0, None)  # ensure non-negative  # ensure non-negative
    pdf = I_flat_pos / I_flat_pos.sum()  # normalize to sum to 1
    cdf = np.cumsum(pdf)  # build CDF
    cdf[-1] = 1.0  # Force exact upper bound — removes all floating point slop

    # Draw uniform samples and find where they land in the CDF
    uniform_samples = 1.0 - rng.uniform(size=N_samples)
    flat_indices = np.searchsorted(cdf, uniform_samples, side="right")  # shape: (N_samples,)

    # Convert flat indices back to 3D grid indices
    ikperp, iphi, ix = np.unravel_index(flat_indices, rad_dist.shape)

    # Look up the corresponding coordinate valuesand add jitter about the bin, so we don't sample exactly on the points
    success = False
    for i in np.arange(0, 1000):
        # Iteratively attempt to jitter around the selected coordinates
        sampled_k_perp = k_perp_values[ikperp] + rng.uniform(-0.5, 0.5, size=N_samples) * dkperp[ikperp]
        sampled_phi = phi_values[iphi] + rng.uniform(-0.5, 0.5, size=N_samples) * dphi[iphi]
        sampled_x = x_values[ix] + rng.uniform(-0.5, 0.5, size=N_samples) * dx[ix]

        # Re-enforce kinematic cuts -- the jitter can take us into unphysical regime -- resample jitter if failed
        if sampled_x < x_min:
            continue
        elif sampled_x > x_max:
            continue
        elif sampled_k_perp < kperp_min:
            continue
        elif sampled_k_perp > kperp_max(sampled_x):
            continue
        else:
            success = True
            break

    # If we failed to successfully sample a momentum, we return nans.
    if not success:
        return np.array([np.nan, np.nan, np.nan])

    # Convert to cartesian momenta
    sampled_kx = sampled_k_perp * np.cos(sampled_phi)
    sampled_ky = sampled_k_perp * np.sin(sampled_phi)
    # Invert $x = omega / E$ for massless particles
    sampled_kz = np.sqrt((sampled_x ** 2) * (E ** 2) - (sampled_k_perp ** 2))

    # Stack cartesian values
    emission_momentum = np.column_stack([sampled_kx, sampled_ky, sampled_kz])

    # Return in appropriate shape
    if N_samples == 1:
        return np.reshape(emission_momentum, 3)
    else:
        return emission_momentum


# Function to return the current emission rate and a sampled emission
def aniso_nn_rate_and_k(particle: hard_particles.Particle, medium: plasma.plasma, dtau: float, nn=None):
    """
    Compute the radiation number distribution and using a neural network model, then

    Params:
        particle: hard_particles.Particle
            The particle to compute the radiation for.
        medium: plasma.plasma
            The surrounding plasma environment.
        dtau: float
            Step size for the evolution in proper time.
        nn: Optional
            The neural network model used to calculate the radiation distribution.

    Returns:
        tuple: Ng (float), k (list)
        - Ng is the total integrated number of gluons expected in the macrostep.
        - k is the sampled kinematics (k_perp, phi, x) of gluon emissions.
    """
    #########################################
    # Gather particle and medium properties #
    #########################################
    if particle.isq:
        CR = 4 / 3
    elif particle.isg:
        CR = 3
    else:
        logging.error("Invalid particle ID for aniso_nn radiation module!")
        raise Exception
    p = particle.p3
    E = np.linalg.norm(p) # Feed massless energy, since using massless derivation
    tau = particle.tau
    point = particle.coords
    temp = medium.temp(point).item()
    mu = mu_DeBye(temp)
    u = np.array(
        [float(medium.x_vel(point).item()), float(medium.y_vel(point).item()), float(medium.z_vel(point).item())])
    uperp = np.linalg.norm(utilities.perp_vec(a=u, b=p))

    #############################################################################################################
    # Compute the extrema of kinematic boundaries on gluon emission and create arrays of the coordinate values. #
    #############################################################################################################
    num_x_points = 20  # number of log-spaced points in kz to compute
    num_k_perp_points = 10  # num of points in kx & ky to compute
    num_phi_points = 32

    # Compute radiation kinematic bounds -- see https://arxiv.org/abs/nucl-th/0112071
    x_min = mu / E  # Minimum x based on minimum plasmon frequency
    x_max = 1 - x_min  # Maximum based on consistency with minimum as an IR regulator
    k_perp_min = mu  # Minimum k_perp based on minimum plasmon frequency
    # Note: maximum of ((Min[x^2, (1-x)^2]  * E^2) - mu^2) --> (E^2 - mu^2) / 4
    k_perp_max = np.sqrt(((E ** 2) - (mu **2)) / 4) # Maxium based on requiring positive z mom. of emission & emitter

    # Create array of points in x
    x_min_pow = np.log10(x_min)  # minimum power of 10 in x to compute
    x_max_pow = np.log10(x_max)  # maximum power of 10 in x to compute
    x_values = np.logspace(x_min_pow, x_max_pow, num_x_points)

    # Create array of points in k_perp
    k_perp_values = np.linspace(k_perp_min, k_perp_max, num_k_perp_points)

    # Create array of points in phi
    phi_values = np.linspace(0, 2 * np.pi, num_phi_points, endpoint=False)  # does not include 2pi -- Overlap w/ 0

    # Create the 3D meshgrid coordinates based on t
    k_perp_grid, _, x_grid = np.meshgrid(k_perp_values, phi_values, x_values, indexing='ij')

    # Get pathlength traveled in this step
    delta_t, delta_x, delta_y, delta_z = particle.next_pathlength(dtau, cart=True)
    delta_pathlength = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

    ####################################################################################
    # Compute the radiation distribution on our desired grid using the neural network. #
    ####################################################################################
    dtau_rad_dist = CR * aniso_rad_dist(E=E, tau=tau, temp=temp, uperp=uperp,
                                        x_values=x_values, k_perp_values=k_perp_values, phi_values=phi_values,
                                        dtau=dtau, nn=nn)

    #################################################################################
    # Integrate the distribution to find the expected number of gluons in this step #
    #################################################################################
    # (np.trapezoid handles integration of arbitrary spacing via coordinates)
    # Note the Jacobian applied to integrate over k_perp and phi as opposed to kx and ky.
    # Integrate over x with trapezoidal method
    integrand_x = np.trapezoid(k_perp_grid * dtau_rad_dist, x_values, axis=2)  # shape: (n_k_perp, n_phi)

    # Integrate over phi via exact uniform Riemann sum (periodic domain,
    # NOT closed at 2*pi -- trapezoid drops the wrap-around segment).
    # This is exact for a truncated Fourier series in cos(phi), cos(2*phi) (our NN output)
    # as long as num_phi_points is not a multiple of 2 (any N > 2 here is fine).
    dphi = 2 * np.pi / num_phi_points
    integrand_x_phi = dphi * np.sum(integrand_x, axis=1)  # shape: (n_k_perp,)

    # Integrate over k_perp (trapezoid appropriate -- non-periodic, linear spacing)
    Ng = np.trapezoid(integrand_x_phi, k_perp_values, axis=0)  # scalar

    ##############################################################################################
    # Sample the distribution for a gluon emission coordinate based on this step's distribution. #
    ##############################################################################################
    if Ng > 0.0:
        k = sample_rad_dist(dtau_rad_dist, N_samples=1,
                            k_perp_values=k_perp_values, phi_values=phi_values,
                            x_values=x_values, mu=mu, E=E)
    else:
        k = np.array([np.nan, np.nan, np.nan])

    # Return your hard earned values, with warnings if something went wrong with the sampling.
    if Ng == 0:
        pass
    elif np.amax(np.isnan(k)):
        logging.warning("Nans in emission momentum!")
        logging.warning(k)
    elif k[2] < 0:
        logging.warning("Backward gluon emission!!!")
    return Ng, k


def aniso_nn(particle: hard_particles.Particle, plasma_object: plasma.plasma, dtau: float, nn):
    """
    Use a Neural Network emulator to compute the radiation spectrum for this particle in this macrostep.

    Perform a simplified Ogata thinning, assuming a constant rate between emissions (within this step).

    Then, sample the kinematics of the emitted particles from the distribution.

    Params:
        particle: hard_particles.Particle
            The particle to compute emissions for.
        plasma_object: plasma.plasma
            The surrounding plasma environment.
        dtau: float
            Macrostep size for evolution in proper time.
        nn: Any
            The neural network model for radiation spectrum computation.

    Returns:
        list: Emitted particle momenta for the macrostep.
    """
    # Check if we should compute the radiation at all
    point = particle.coords
    temp = plasma_object.temp(point).item()
    if temp == np.nan:  # Cancel evolution if we exit the plasma space
        raise NoMedium()
    elif temp < config.jet.T_HRG:  # Cancel evolution if we exit the plasma phase
        raise HadronGas()

    # Storage array for emissions
    emission_momenta = []

    # Create a subdivision counter to march through this step, sampling the position of the next emission based on the
    # radiation rate. Exit when we either finish the step or the particle reaches our desired medium scale.
    tau_0 = particle.tau
    tau_f = tau_0 + dtau
    tau = tau_0
    total_kz = 0
    particle_p_norm = np.linalg.norm(particle.p3)
    while total_kz < particle_p_norm - config.jet.EMIN:
        # Compute radiation distribution from this macrostep, integrated number of expected gluons, and a sampled k
        # distribution returned in (k_perp, phi, x) in parton frame,
        Ng, k = aniso_nn_rate_and_k(particle=particle, medium=plasma_object, dtau=dtau, nn=nn)

        # If Ng invalid or zero, break
        if Ng == 0:
            break
        elif np.isnan(Ng):
            logging.warning(f"NaN expected number of emissions: {Ng}")
            break
        elif Ng < 0.0:
            logging.warning(f"Negative expected number of emissions: {Ng}")
            break

        # Approximate radiation rate
        rate = Ng / dtau
        # logging.debug(f"Ratio over analytic: {rate / (N_gluons(particle=particle, medium=plasma_object, dtau=dtau) / dtau)}")

        # Survival probability style sample for next emission position -- no thinning for constant known rate
        tau_to_emit = rng.exponential(1.0 / rate)
        next_tau = tau + tau_to_emit

        # If the next emission lies outside this macro timestep, break and stop emitting in this step
        if next_tau > tau_f:
            break

        # Otherwise, accept the emission
        logging.debug(f"Emitting gluon! Radiation frame momentum:")

        # If we rescale energies, do it!
        # Compute expected energy of emitted gluons
        logging.debug(f"k = {k} GeV")

        # Check if gluon is backward facing -- This shouldn't happen from a single step's radiation,
        # but summing the distribution over multiple steps "turns" the coordinate system such that
        # it has a nonzero total probability
        if k[2] < 0.0:
            logging.warning(
                "!\n!\n!\nEmitted gluon is backward facing. Not good!\n!\n!\n!")

        # Transform emission momentum to lab frame
        k = lf_emission_momentum(k=k, particle=particle, medium=plasma_object)
        logging.debug(f"Lab frame momentum:")
        logging.debug(f"k = {k} GeV")
        logging.debug(f"p = {particle.p3} GeV")

        # Append momenta to list for this step
        emission_momenta.append(k)
        total_kz += np.linalg.norm(k)

        # Step forward in time and reiterate
        tau = next_tau

    # Return all of the emissions for this macrostep
    return emission_momenta