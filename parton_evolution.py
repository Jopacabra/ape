import logging
import numpy as np
from scipy.special import keip_zeros, kerp_zeros
from typing_extensions import NoDefault

import plasma
import hard_particles
import plasma_interaction
import plasma_interaction as pi
import config
from plasma_interaction import NoMedium
import utilities
import time
import math


"""
This module takes a single hard_particles.Particle object and evolves it throughout the plasma phase of a 
plasma.Plasma object.

If a tau value is provided, the particle will be evolved by that amount of time. Otherwise, it will be evolved to the end of the 
plasma.

Returns a list of emmitted particle momenta, plus
True if the particle was evolved the full tau window requested or false if it was not.
"""
def evolve_particle(particle : hard_particles.Particle, plasma_object : plasma.plasma, nn=None,
                    rng=np.random.default_rng(), tau=None):
    # Create list of emitted particles to be tracked later
    emission_momenta_total = []
    emission_coords_total = []

    #########################
    # Perform the evolution #
    #########################

    try:
        # Read settings and object properties
        dtau = config.jet.DTAU
        tau_0 = particle.tau
        if tau is None:
            tau_f = plasma_object.tf
        else:
            tau_f = tau_0 + tau

        # Freestream until thermalization of medium using very small steps
        while particle.tau < plasma_object.t0:
            """
            It is extremely important that our step size is very small in the early time, elsewise you get very
            upsetting zig-zagging numerical artifacts in the trajectory due to 1/tau being very large. Particles will
            "overshoot" their target rapidity (by a lot), then eventually saturate to their proper final trajectory.
            
            It is reasonable to make this value of order the hard scattering time config.jet.TAU_PROD, so that is what
            we lock it to. This avoids user error in setting the production time even smaller than this value.
            """
            particle.prop(dtau=config.jet.TAU_PROD)

        # Compute remaining number of steps to evolve
        num_steps = int((tau_f - particle.tau)/dtau)

        # Quarks and gluons have medium interaction
        if particle.isq or particle.isg:

            steps_complete = 0
            total_energy = 0  # Counter for storing expected energy of emissions
            for step_i in range(num_steps):

                ########################
                # Thermalization Check #
                ########################
                # Particles too close to the medium scale can't be treated perturbatively...
                # We resample a momentum from a thermal distribution and freestream.
                if particle.E < config.jet.EMIN:
                    logging.debug(
                        "Particle energy approaching medium scale. Sampling final momentum from thermal dist...")
                    particle.thermal_sample()  # Samples and sets in-place a final momentum from a thermal distribution
                    return emission_momenta_total, emission_coords_total, False  # Don't bother to freestream -- save time

                #############
                # Radiative #
                #############

                # Compute simple GLV energy loss
                try:
                    t0 = time.time()
                    if config.jet.RAD_MODEL == "aniso_NN":
                        """
                        Use a Neural Network emulator to commute the radiation spectrum for this particle in this step.
                        
                        Poisson sample about the integral of the radiation spectrum to determine the number of gluons 
                        to emit. Then, sample the kinematics of the particles from the distribution. Finally, rescale 
                        the emissions to the expected energy loss.
                        """
                        # Hardcoded options
                        fixed_norm = True  # Overwrite the normalization of the distribution with analytic estimate.
                        poisson_E = False  # Poisson sample the energy of each emission

                        # Compute radiation kinematic bounds -- see https://arxiv.org/abs/nucl-th/0112071
                        point = particle.coords
                        temp = plasma_object.temp(point).item()
                        mu = pi.mu_DeBye(temp)
                        x_min = mu / (2 * particle.E0)
                        x_max = 1 - x_min

                        # Create bins of kz
                        x_min_pow = np.log10(x_min)  # minimum power of 10 in x to compute
                        x_max_pow = np.log10(x_max)  # maximum power of 10 in x to compute
                        num_k_points = 20  # number of log-spaced points in kz to compute
                        x_values = np.logspace(x_min_pow, x_max_pow, num_k_points // 2)
                        k_pos_values = x_values * particle.E0
                        k_values = k_pos_values  # no need for negative kz now!

                        # Create bins in k_perp
                        # Note: maximum of ((Min[x^2, x(1-x)] * 4 * E_0^2) - mu^2) --> E_0^2 - mu^2
                        # Added a factor of 0.25, because everything else is expensive numerically small probabilities
                        # !!!!!!!!!!!!! Revisit this later !!!!!!!!!!!!!
                        num_k_perp_points = 25  # num or (num - 1) of points in kx & ky to compute -- 0 added
                        k_perp_pos_values = np.linspace(0, 0.25 * np.sqrt(particle.E0 ** 2 - mu ** 2),
                                                        num_k_perp_points // 2)
                        k_perp_values = np.concatenate((-np.flip(k_perp_pos_values[1:]), k_perp_pos_values))

                        _, _, x_grid = np.meshgrid(k_perp_values, k_perp_values, x_values, indexing='ij')

                        emission_momenta = []

                        # Compute radiation distribution from this step -- returned in (kx, ky, kz) in parton frame
                        dtau_rad_dist = pi.aniso_rad_dist(particle=particle, medium=plasma_object, dtau=dtau,
                                                                          kx_values=k_perp_values,
                                                                          ky_values=k_perp_values,
                                                                          kz_values=k_values, nn=nn)

                        # Compute integral of complete radiation distribution
                        # (np.trapezoid handles integration of arbitrary spacing via coordinates)
                        if fixed_norm:
                            # Fix analytic expectation for gluon emissions per step
                            total_number = pi.N_gluons(particle=particle, medium=plasma_object, dtau=dtau)
                            total_energy += pi.E_gluons(particle=particle, medium=plasma_object, dtau=dtau)
                        else:
                            total_number = np.trapezoid(
                                np.trapezoid(
                                    np.trapezoid(dtau_rad_dist, k_values, axis=2),  # Integrate over kz -> shape: (n_kx, n_ky)
                                                          k_perp_values, axis=1),  # Integrate over ky -> shape: (n_kx)
                                                          k_perp_values, axis=0)  # Integrate over kx -> scalar
                            total_energy += np.trapezoid(
                                np.trapezoid(
                                    np.trapezoid(dtau_rad_dist * x_grid * particle.E0, k_values, axis=2),  # Integrate over kz -> shape: (n_kx, n_ky)
                                                          k_perp_values, axis=1),  # Integrate over ky -> shape: (n_kx)
                                                          k_perp_values, axis=0)  # Integrate over kx -> scalar
                            logging.debug(f"Radiation number distribution integral: {total_number}")
                            logging.debug(f"Radiation energy distribution integral: {total_energy}")

                        # Poisson sample to determine number of gluons to emit, with an average of this step's number
                        n = rng.poisson(lam=total_number, size=1).item()

                        # Sample and rescale emission kinematics
                        for i in range(n):
                            # Sample the distribution for emission kinematics in the radiation frame
                            k = plasma_interaction.sample_rad_dist(dtau_rad_dist, N_samples=1,
                                                                   kx_values=k_perp_values, ky_values=k_perp_values,
                                                                   kz_values=k_values, mu=mu, E=particle.E0)
                            logging.debug(f"Emitting gluon! Radiation frame info:")

                            # If we rescale energies, do it!
                            # Compute expected energy of emitted gluons
                            logging.debug(f"p = {particle.p3} GeV")
                            logging.debug(f"k = {k} GeV")
                            if fixed_norm:
                                # Use analytic expectation
                                expected_E = total_energy / n

                                # Poisson sample, if you want -- Doesn't really make sense to do a discrete sample...
                                if poisson_E:
                                    expected_E = rng.poisson(lam=expected_E, size=1).item()

                                # Don't allow emission of higher energy than particle's current energy.
                                if expected_E > particle.E:
                                    logging.warning(
                                        f"Expected emission energy {expected_E} GeV is higher than particle's current energy {particle.E} GeV.")
                                    expected_E = particle.E  # Particle should thermalize on the next step.

                                # Set longitudinal momentum to expected energy loss. k_perp does not reduce E.
                                k[2] = expected_E
                                if not math.isclose(k[2], expected_E, rel_tol=1e-12):
                                    logging.warning(
                                        f"Rescaled gluon energy expected {expected_E} GeV, but {np.linalg.norm(k)} GeV")

                            # Check if gluon is backward facing -- This shouldn't happen from a single step's radiation,
                            # but summing the distribution over multiple steps "turns" the coordinate system such that
                            # it has a nonzero total probability
                            if k[2] < 0.0:
                                logging.warning(
                                    "!\n!\n!\nEmitted gluon is backward facing. Not good!\n!\n!\n!")


                            # Transform emission momentum to lab frame
                            k = pi.lf_emission_momentum(k=k, particle=particle, medium=plasma_object)
                            logging.debug(f"k = {k} GeV")

                            # Append momenta to list for this step
                            emission_momenta.append(k)

                        # Find the total momentum of the emitted particles
                        if n == 0:
                            # No emission, so set radiation momentum to zero
                            total_k = np.array([0, 0, 0])
                        else:
                            # Sum emission momenta from this step
                            total_k = np.sum(emission_momenta, axis=0)

                            # Only reset if this energy was "given a chance" to emit
                            total_energy = 0

                        # Create particle delta opposite to the total emitted gluon momentum in the lab coord. system
                        rad_delta = hard_particles.ParticleDelta(dpx=-total_k[0], dpy=-total_k[1], dpz=-total_k[2])

                        # Append momenta and coords to complete evolution list
                        for i in np.arange(len(emission_momenta)):
                            emission_momenta_total.append(emission_momenta[i])


                    elif config.jet.RAD_MODEL == "iso_analytic":
                        rad_delta = pi.rad_delta(particle, plasma_object, dtau)
                    elif config.jet.RAD_MODEL == "None":
                        rad_delta = hard_particles.ParticleDelta(dpx=0, dpy=0, dpz=0)
                    else:
                        logging.error("Unknown RAD_MODEL, defaulting to iso_analytic")
                        rad_delta = pi.rad_delta(particle, plasma_object, dtau)
                except pi.HadronGas:
                    logging.debug("Particle escaped plasma.")
                    break
                except NoMedium:
                    logging.debug("Particle escaped plasma grid.")
                    break
                except Exception as e:
                    logging.debug("Exception occurred at step {}".format(step_i))
                    logging.exception(e)
                    rad_delta = hard_particles.ParticleDelta(dpx=0, dpy=0, dpz=0)
                    break

                dt = time.time() - t0
                logging.debug(f"Radiation computed in {dt}s")

                ###############
                # Collisional #
                ###############
                coll_t0 = time.time()
                try:
                    if config.jet.COL_MODEL == "flow":
                        # logging.warning("FLOW!")
                        coll_delta = pi.collisional_delta(particle, plasma_object, dtau)
                    elif config.jet.COL_MODEL == "flowgrad":
                        # logging.warning("FLOWGRAD!")
                        # Use linear gradients for collisional interaction
                        coll_delta = pi.collisional_delta_linear_gradients(particle, plasma_object, dtau)
                    else:
                        logging.warning("Invalid collisional model choice. No collisional interactions.")
                        coll_delta = hard_particles.ParticleDelta(dpx=0, dpy=0, dpz=0)

                except pi.HadronGas:
                    logging.debug("Particle escaped plasma.")
                    break
                except NoMedium:
                    logging.debug("Particle escaped plasma grid.")
                    break
                except Exception as e:
                    logging.debug("Exception occurred at step {}".format(step_i))
                    logging.exception(e)
                    coll_delta = hard_particles.ParticleDelta(dpx=0, dpy=0, dpz=0)
                    break
                coll_dt = time.time() - coll_t0
                # logging.debug(f"Collisional interaction computed in {coll_dt}s")

                #######################
                # Propagate particle  #
                #######################

                # Step forward with original momentum
                particle.prop(dtau=dtau)
                steps_complete += 1  # Mark step as complete -- particle has moved forward

                # Add interaction momentum
                particle.apply_deltap(coll_delta)
                particle.apply_deltap(rad_delta)

                # Set emission coordinates for this step to the position AFTER propagation, right where momentum changes
                try:
                    for i in np.arange(len(emission_momenta)):
                        emission_coords_total.append(particle.coords)
                except:
                    pass

            # Freestream any remaining evolution time
            if steps_complete < num_steps:
                logging.debug("Freestreaming {} steps...".format(num_steps - steps_complete))
                for step_j in range(steps_complete, num_steps):
                    particle.prop(dtau=dtau)

            # Notify of remaining radiation potential
            try:
                # logging.debug(f"{(total_number)} unradiated gluons remaining")
                pass
            except:
                pass

            return emission_momenta_total, emission_coords_total, True

        # EW Bosons have no medium interaction, so they should freestream through the medium
        elif particle.isEWB:
            logging.debug("Electroweak boson. Freestreaming {} steps...".format(num_steps))
            for step_i in range(num_steps):
                # Propagate particle
                particle.prop(dtau=dtau)

            return emission_momenta_total, emission_coords_total, True

        else:
            return emission_momenta_total, emission_coords_total, False


    except hard_particles.StopEvolve as e:
        logging.debug(e)
        return emission_momenta_total, emission_coords_total, False

    except Exception as e:
        logging.exception(e)
        logging.error(e)
        return emission_momenta_total, emission_coords_total, False


