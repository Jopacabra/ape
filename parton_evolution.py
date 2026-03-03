import logging
import numpy as np

import plasma
import hard_particles
import plasma_interaction
import config

"""
This module takes a single hard_particles.Particle object and evolves it throughout the plasma phase of a 
plasma.Plasma object.

If a tau value is provided, the particle will be evolved by that amount of time. Otherwise, it will be evolved to the end of the 
plasma.
"""
def evolve_particle(particle : hard_particles.Particle, plasma_object : plasma.plasma_event, tau=None):
    logging.info('Evolving particle {}...'.format(particle.to_kwargs()))
    try:
        # Read settings and object properties
        dtau = config.jet.DTAU
        tau_0 = particle.tau
        if tau is None:
            tau_f = plasma_object.tf
        else:
            tau_f = tau_0 + tau

        # Freestream until thermalization
        while particle.tau < plasma_object.t0:
            particle.prop(dtau=dtau)

        # Compute number of steps to evolve
        num_steps = int((tau_f - particle.tau)/dtau)

        # Far forward or backward rapidity particles can't be reasonably treated with our boost-invariance 2+1D medium.
        if np.abs(particle.rap) > 2:
            logging.debug("Off-mid rapidity. Freestreaming {} steps...".format(num_steps))
            for step_i in range(num_steps):
                # Propagate particle
                particle.prop(dtau=dtau)
                continue

        # Quarks and gluons have medium interaction
        if particle.isq or particle.isg:
            steps_complete = 0
            for step_i in range(num_steps):

                #############
                # Radiative #
                #############

                # Compute simple GLV energy loss
                try:
                    rad_delta = plasma_interaction.rad_delta(particle, plasma_object, dtau)
                except plasma_interaction.HadronGas:
                    logging.info("Particle escaped plasma.")
                    break
                except plasma.NoMedium:
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
                except plasma_interaction.HadronGas:
                    logging.info("Particle escaped plasma.")
                    break
                except plasma.NoMedium:
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

    except Exception as e:
        logging.exception(e)
        logging.error(e)
        return False