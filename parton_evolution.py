import logging
import numpy as np
from typing_extensions import NoDefault

import plasma
import hard_particles
import plasma_interaction
import config
from plasma_interaction import NoMedium
import utilities
import time


"""
This module takes a single hard_particles.Particle object and evolves it throughout the plasma phase of a 
plasma.Plasma object.

If a tau value is provided, the particle will be evolved by that amount of time. Otherwise, it will be evolved to the end of the 
plasma.

Returns a list of emmitted particle momenta, plus
True if the particle was evolved the full tau window requested or false if it was not.
"""
def evolve_particle(particle : hard_particles.Particle, plasma_object : plasma.plasma_event, tau=None):
    # Create list of emitted particles to be tracked later
    emission_momenta_total = []
    emission_coords_total = []

    # Start Radiation Tracker, if necessary
    if config.jet.RAD_MODEL == "aniso_NN":
        x_points = 10  # Number of log-spaced points in x to compute
        k_points = 50  # Even number of lin-spaced points in kx and ky to compute
        max_kx_ky = 0.05*particle.E0  # Maybe should be dependent on energy, needs testing.
        x_values = np.logspace(-4, 0, x_points)
        kx_values = np.linspace(-max_kx_ky, max_kx_ky, k_points)
        ky_values = np.linspace(-max_kx_ky, max_kx_ky, k_points)
        rad_dist = np.zeros(shape=(x_points, k_points, k_points))

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
                        emission_coords = []
                        dtau_rad_dist = plasma_interaction.aniso_rad_dist(particle=particle, medium=plasma_object, dtau=dtau,
                                                                    kx_values=kx_values, ky_values=ky_values[int(k_points / 2)::],
                                                                    x_values=x_values, nn=nn)

                        # Rotate to absolute coordinates
                        """
                        We start with a distribution in the "parton frame", where x is aligned with u_perp, z is aligned 
                        with the hard particle momentum, and y = z (cross) x.
                        
                        We want to rotate this distribution to the lab frame, so we can sum multiple distributions from 
                        different frames.
                        """


                        # Add rotated step distribution to the total distribution
                        rad_dist = rad_dist + dtau_rad_dist
                        dt = time.time() - t0
                        logging.debug(f"Radiation computed in {dt}s")

                        # Compute integral of complete radiation distribution
                        N_kx_x = np.trapezoid(rad_dist, ky_values, axis=2)  # Integrate over ky -> shape: (n_x, n_kx)
                        N_x = np.trapezoid(N_kx_x, kx_values, axis=1)  # Integrate over kx -> shape: (n_x,)
                        # x integration in log-space (accounts for log-spaced grid) -- includes Jacobian, factor of x
                        total_integral = np.trapezoid(N_x * x_values, np.log(x_values))  # Integrate over x -> scalar

                        # Check if we should emit a particle
                        N_norm = 1
                        total_k = np.array([0, 0, 0])
                        current_integral = total_integral
                        n = 0
                        while current_integral >= N_norm:
                            n += 1
                            logging.debug("Emitting gluon")
                            # Sample the distribution for emission kinematics in the radiation frame
                            k = utilities.sample_rad_dist(rad_dist, E=particle.E0, N_samples=1,
                                                          kx_values=kx_values, ky_values=ky_values, x_values=x_values)
                            coords = particle.coords

                            # Transform emission momentum to lab frame
                            logging.debug(k)
                            k = utilities.lf_emission_momentum(k=k, particle=particle, medium=plasma_object)
                            logging.debug(k)

                            # Append momenta and coords to list for this step
                            emission_momenta.append(k)
                            emission_coords.append(coords)

                            # Append momenta and coords to complete evolution list
                            emission_momenta_total.append(k)
                            emission_coords_total.append(coords)

                            # Reduce number of total emissions remaining by 1
                            current_integral -= N_norm

                        # Rescale distribution, removing "n gluons" of emission probability
                        rad_dist = ((total_integral - n*N_norm) / total_integral) * rad_dist

                        if n == 0:
                            # No emission, so set radiation momentum to zero
                            total_k = np.array([0, 0, 0])
                        else:
                            # Sum emission momenta from this step
                            total_k = np.sum(emission_momenta, axis=0)

                        # Create particle delta opposite to the emitted particle momentum, in the lab coordinate system
                        rad_delta = hard_particles.ParticleDelta(dpx=-total_k[0], dpy=-total_k[1], dpz=-total_k[2])


                    elif config.jet.RAD_MODEL == "iso_analytic":
                        rad_delta = plasma_interaction.rad_delta(particle, plasma_object, dtau)
                    else:
                        logging.error("Unknown RAD_MODEL, defaulting to iso_analytic")
                        rad_delta = plasma_interaction.rad_delta(particle, plasma_object, dtau)
                except plasma_interaction.HadronGas:
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
                try:
                    coll_delta = plasma_interaction.collisional_delta(particle, plasma_object, dtau)
                    # # Use linear gradients for collisional interaction
                    # coll_delta = plasma_interaction.collisional_delta_linear_gradients(particle, plasma_object, dtau)

                except plasma_interaction.HadronGas:
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

                #######################
                # Propagate particle  #
                #######################

                # Step forward with original momentum
                particle.prop(dtau=dtau)
                steps_complete += 1  # Mark step as complete -- particle has moved forward

                # Add interaction momentum
                particle.apply_deltap(coll_delta)
                particle.apply_deltap(rad_delta)

            # Freestream any remaining evolution time
            if steps_complete < num_steps:
                logging.debug("Freestreaming {} steps...".format(num_steps - steps_complete))
                for step_j in range(steps_complete, num_steps):
                    particle.prop(dtau=dtau)

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


