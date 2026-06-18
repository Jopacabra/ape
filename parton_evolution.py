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

    # Start Radiation Tracker, if necessary
    if config.jet.RAD_MODEL == "aniso_NN":
        x_min = -2  # minimum power of 10 in x to compute
        kz_points = 50  # Number of evenly spaced points in x to compute
        kperp_points = 50  # Even number of lin-spaced points in kx and ky to compute
        x_values = np.logspace(x_min, -0.05, kperp_points//2)
        k_pos_values = x_values*particle.E0
        # k_pos_values = np.linspace(0.01, 1, kperp_points // 2)*particle.E0
        k_values = np.concatenate((-k_pos_values, k_pos_values))

        # Find energy of emission at each coordinate
        kxkx, kyky, kzkz = np.meshgrid(k_values, k_values, k_values, indexing='ij')
        E_values = np.sqrt(kxkx ** 2 + kyky ** 2 + kzkz ** 2)

        # Construct zeroed radiation distribution array
        rad_dist = np.zeros(shape=(kperp_points, kperp_points, kz_points))
        total_number = 0
        total_energy = 0

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
            for step_i in range(num_steps):

                ########################
                # Thermalization Check #
                ########################
                # Particles too close to the medium scale can't be treated perturbatively...
                # We resample a momentum from a thermal distribution and freestream.
                if particle.E < config.jet.EMIN:
                    logging.debug(
                        "Particle energy approaching medium scale. Sampling final momentum from thermal dist...")
                    particle.thermal_sample()  # Samples and sets a final momentum from a thermal distribution
                    return emission_momenta_total, emission_coords_total, False  # Don't bother to freestream -- save time

                #############
                # Radiative #
                #############

                # Compute simple GLV energy loss
                try:
                    if config.jet.RAD_MODEL == "aniso_NN":
                        t0 = time.time()

                        emission_momenta = []

                        # Compute radiation distribution from this step -- returned in (kx, ky, kz) in parton frame
                        dtau_rad_dist = pi.aniso_rad_dist(particle=particle, medium=plasma_object, dtau=dtau,
                                                                          kx_values=k_values, ky_values=k_values,
                                                                          kz_values=k_values, nn=nn)



                        # # Rotate to absolute coordinates
                        # """
                        # We start with a distribution in the "parton frame", where x is aligned with u_perp, z is aligned
                        # with the hard particle momentum, and y = z (cross) x.
                        #
                        # We want to rotate this distribution to the lab frame, so we can sum multiple distributions from
                        # different frames.
                        #
                        # We do this with an image processing interpolation. The coordinates of the new grid are the same
                        # k_values, ky_values, and kz_values arrays -- we just understand them to be in the lab frame.
                        # """
                        # dtau_rad_dist = pi.rotate_rad_dist(particle=particle, medium=plasma_object,
                        #                                                    rad_dist=dtau_rad_dist,
                        #                                                    kx_values=k_values,
                        #                                                    ky_values=k_values,
                        #                                                    kz_values=k_values)


                        # Add rotated step distribution to the total distribution
                        rad_dist = rad_dist + dtau_rad_dist
                        dt = time.time() - t0
                        logging.debug(f"Radiation distribution computed in {dt}s")

                        # Compute integral of complete radiation distribution
                        # (np.trapezoid handles integration of arbitrary spacing via coordinates)
                        t0 = time.time()
                        fixed_gluons = True
                        poisson_N = True
                        poisson_E = True
                        if fixed_gluons:
                            """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
                            # Fix analytic expectation for gluon emissions per step
                            total_number += pi.N_gluons(particle=particle, medium=plasma_object, dtau=dtau)
                            total_energy += pi.E_gluons(particle=particle, medium=plasma_object, dtau=dtau)
                            """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
                        else:
                            total_number += np.trapezoid(
                                np.trapezoid(
                                    np.trapezoid(dtau_rad_dist, k_values, axis=2),  # Integrate over kz -> shape: (n_kx, n_ky)
                                                          k_values, axis=1),  # Integrate over ky -> shape: (n_kx)
                                                          k_values, axis=0)  # Integrate over kx -> scalar
                            total_energy += np.trapezoid(
                                np.trapezoid(
                                    np.trapezoid(dtau_rad_dist * x_values * particle.E0, k_values, axis=2),  # Integrate over kz -> shape: (n_kx, n_ky)
                                                          k_values, axis=1),  # Integrate over ky -> shape: (n_kx)
                                                          k_values, axis=0)  # Integrate over kx -> scalar
                            logging.debug(f"Radiation number distribution integral: {total_number}")
                            logging.debug(f"Radiation energy distribution integral: {total_energy}")

                        # Check how many gluons to emit this step
                        N_norm = 1  # Enhancement on number of gluons, for forcing emission in debug
                        if poisson_N:
                            # Poisson sample to determine number of gluons to emit, with an average of this step's number
                            print(total_number)
                            n = rng.poisson(lam=N_norm*total_number, size=1).item()
                        else:
                            # Wait until we accumulate "1 gluon" worth of emissions before emitting
                            n = total_number // N_norm


                        # Sample and rescale emission kinematics
                        for i in range(n):
                            # Sample the distribution for emission kinematics in the radiation frame
                            k = plasma_interaction.sample_rad_dist(rad_dist, N_samples=1,
                                                                   kx_values=k_values, ky_values=k_values,
                                                                   kz_values=k_values)
                            logging.debug(f"Emitting gluon! Radiation frame info:")

                            # If we rescale energies, do it!
                            # Compute expected energy of emitted gluons
                            logging.debug(f"p = {particle.p3} GeV")
                            logging.debug(f"k = {k} GeV")
                            if fixed_gluons:
                                # Use analytic expectation
                                expected_E = total_energy / n

                                # Poisson sample, if you want
                                if poisson_E:
                                    expected_E = rng.poisson(lam=expected_E, size=1).item()

                                # Don't allow emission of higher energy than particle's current energy.
                                if expected_E > particle.E:
                                    logging.warning(
                                        f"Expected emission energy {expected_E} GeV is higher than particle's current energy {particle.E} GeV.")
                                    expected_E = particle.E  # Particle should thermalize on the next step.

                                # Set longitudinal momentum to expected energy loss. k_perp does not reduce E.
                                k[2] = expected_E
                                logging.debug(f"k = {k} GeV")
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

                            # Append momenta to list for this step
                            emission_momenta.append(k)

                        # Rescale radiation distribution to account for number of gluons that we emitted
                        if poisson_N:
                            # Kill the entire number distribution -- these gluons were "given a chance" to emit
                            rad_dist = np.zeros_like(rad_dist)

                            # Reset total number of emissions and energy remaining
                            total_number = 0

                            # Reset total energy remaining, if we emitted
                            if n > 0:
                                # Only reset if this energy was "given a chance" to emit
                                total_energy = 0
                        else:
                            # If we emitted, Rescale distribution, removing "n gluons" of emission probability
                            if n > 0:
                                rad_dist = ((total_number - n * N_norm) / total_number) * rad_dist
                                total_number -= n * N_norm

                        # Find the total momentum of the emitted particles
                        if n == 0:
                            # No emission, so set radiation momentum to zero
                            total_k = np.array([0, 0, 0])
                        else:
                            # Sum emission momenta from this step
                            total_k = np.sum(emission_momenta, axis=0)

                        # Create particle delta opposite to the total emitted gluon momentum in the lab coord. system
                        rad_delta = hard_particles.ParticleDelta(dpx=-total_k[0], dpy=-total_k[1], dpz=-total_k[2])

                        # Append momenta and coords to complete evolution list
                        for i in np.arange(len(emission_momenta)):
                            emission_momenta_total.append(emission_momenta[i])

                        dt = time.time() - t0
                        logging.debug(f"Emissions computed in {dt}s")


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


                ###############
                # Collisional #
                ###############
                coll_t0 = time.time()
                try:
                    coll_delta = pi.collisional_delta(particle, plasma_object, dtau)
                    # # Use linear gradients for collisional interaction
                    # coll_delta = pi.collisional_delta_linear_gradients(particle, plasma_object, dtau)

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


