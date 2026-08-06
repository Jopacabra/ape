import logging
import numpy as np
from scipy.special import keip_zeros, kerp_zeros
from typing_extensions import NoDefault

import plasma
import hard_particles
import plasma_interaction
import plasma_interaction as pi
import config
from plasma_interaction import NoMedium, rad_energy_integrand
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
        if num_steps < 1:
            logging.warning("Particle already evolved beyond tau_f. No evolution to perform.")
            return emission_momenta_total, emission_coords_total, False

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
                    rad_t0 = time.time()
                    if config.jet.RAD_MODEL == "aniso_NN":
                        # Compute emissions from NN method
                        emission_momenta = pi.aniso_nn(particle=particle, plasma_object=plasma_object, dtau=dtau, nn=nn)

                        # Find the total momentum of the emitted particles
                        n = len(emission_momenta)
                        if n == 0:
                            # No emission, so set radiation momentum to zero
                            total_k = np.array([0, 0, 0])
                        else:
                            # Sum emission momenta from this step
                            total_k = np.sum(emission_momenta, axis=0)

                            # Append momenta and coords to complete evolution list
                            for i in np.arange(len(emission_momenta)):
                                emission_momenta_total.append(emission_momenta[i])

                        # Create particle delta opposite to the total emitted gluon momentum in the lab coord. system
                        rad_delta = hard_particles.ParticleDelta(dpx=-total_k[0], dpy=-total_k[1], dpz=-total_k[2])


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

                rad_dt = time.time() - rad_t0
                logging.debug(f"Radiation computed in {rad_dt}s")

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


