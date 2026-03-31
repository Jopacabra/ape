import logging
import numpy as np

import plasma
import hard_particles
import plasma_interaction
import config
from plasma_interaction import NoMedium

"""
This module takes a single hard_particles.Particle object and evolves it throughout the plasma phase of a 
plasma.Plasma object.

If a tau value is provided, the particle will be evolved by that amount of time. Otherwise, it will be evolved to the end of the 
plasma.

Returns True if the particle was evolved the full tau window requested. Returns false if it was not.
"""
def evolve_particle(particle : hard_particles.Particle, plasma_object : plasma.plasma_event, tau=None):
    logging.info('Evolving particle {}...'.format(particle.to_kwargs()))

    #####################################
    # Choose if we evolve this particle #
    #####################################
    # Far forward or backward rapidity particles can't be reasonably treated with our boost-invariance 2+1D medium.
    if np.abs(particle.rap) > config.jet.RAP_MAX_EVOLVE:
        logging.debug("Large rapidity. Skipping particle...")
        return False
    if not particle.isg and not particle.isq and not particle.isEWB:
        logging.debug("Untreated particle. Skipping particle...")
        return False

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
                    return False  # Don't bother to freestream -- save time

                #############
                # Radiative #
                #############

                # Compute simple GLV energy loss
                try:
                    rad_delta = plasma_interaction.rad_delta(particle, plasma_object, dtau)
                except plasma_interaction.HadronGas:
                    logging.info("Particle escaped plasma.")
                    break
                except NoMedium:
                    logging.info("Particle escaped plasma grid.")
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
                    logging.info("Particle escaped plasma.")
                    break
                except NoMedium:
                    logging.info("Particle escaped plasma grid.")
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
                logging.info("Freestreaming {} steps...".format(num_steps - steps_complete))
                for step_j in range(steps_complete, num_steps):
                    particle.prop(dtau=dtau)

            return True

        # EW Bosons have no medium interaction, so they should freestream through the medium
        elif particle.isEWB:
            logging.debug("Electroweak boson. Freestreaming {} steps...".format(num_steps))
            for step_i in range(num_steps):
                # Propagate particle
                particle.prop(dtau=dtau)

            return True

        else:
            return False


    except hard_particles.StopEvolve as e:
        logging.debug(e)
        return False

    except Exception as e:
        logging.exception(e)
        logging.error(e)
        return False


