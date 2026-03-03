
import logging
import sys
import timeit

import numpy as np
import matplotlib.pyplot as plt

import pythia
import plasma
import hard_particles
import parton_evolution
import collision
import observables


############
# Settings #
############
analyze = True
theta_bins = np.linspace(-1, 1, 21)
EECs = np.zeros(len(theta_bins)-1)

visualize = False
num_events = 100

#############################
# Logging and File Handling #
#############################
# Clear any existing logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create log file & configure logging to be handled into the file AND stderr
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('log.log')
    ],
    force=True
)

# Ignore obnoxious font-search problems
logging.getLogger('matplotlib.font_manager').disabled = True


########################
# Soft Event Evolution #
########################

"""
Create or load a single simple event.
"""
# logging.info('Creating simple plasma...')
# # plasma_object = collision.woods_saxon_plasma(b=5.5, T0=0.39, A=208, a=0.546, alpha=1, name=None,
# #                        resolution=5, rmax=10, tmin=0.5, tmax=None, umax=1, return_grids=False)
# plasma_object = collision.gaussian_plasma(b=5, T0=0.39, A=208, alpha=0.2, name=None,
#                        resolution=5, rmax=10, tmin=0.5, tmax=None, umax=0.75, return_grids=False)
# logging.info('Simple plasma created.')

logging.info("Loading Duke Average Plasma...")
plasma_file = plasma.osu_hydro_file("stored_events/Duke_avg/event_0/viscous_14_moments_evo.dat")
plasma_object = plasma.plasma_event(hydro_object=plasma_file)
logging.info("Duke Average Plasma Loaded.")


########################
# Hard Event Evolution #
########################

hard_event_records = np.array([])
for i in range(num_events):
    ##################
    # Jet Production #
    ##################

    """
    Get a hard particle event from Pythia.
    """
    # Production point
    logging.info('Getting hard scattering...')
    tau_0 = 0.01
    x_0 = 1
    y_0 = 1
    etas_0 = 0.0
    hard_event = pythia.scattering(tau=tau_0, x=x_0, y=y_0, etas=etas_0)
    num_hard_particles = len(hard_event.particles)
    logging.info('Hard scattering done.')


    #################
    # Jet Evolution #
    #################

    """
    Perform the evolution on each particle
    """
    logging.info('Evolving particles...')
    for particle in hard_event.particles:
        parton_evolution.evolve_particle(particle, plasma_object)
    logging.info('Particle evolution complete')


    #################
    # Hadronization #
    #################

    """
    Hadronize hard particles using Lund-String hadronization.
    """
    hard_event_hadrons = pythia.pp_shower_hadronize(hard_event)


    ##########################
    # Optional visualization #
    ##########################
    if visualize:
        logging.info('Visualizing...')

        hard_event.plot_trajectories(z_axis=None, rap_max=5)
        hard_event.plot_trajectories(z_axis="z", rap_max=5)
        # hard_event.plot_trajectories(z_axis="etas", rap_max=5)
        # plt.savefig("particle_trajectories.png", dpi=150)

    #####################
    # Optional analysis #
    #####################
    if analyze:
        logging.info('Analyzing...')
        current_EECs, _ = observables.EEC(hard_event_hadrons, plot=False, bins=theta_bins)

        EECs = EECs + current_EECs


###########################
# Post Evolution Analysis #
###########################

if analyze:
    plt.figure(figsize=(8, 6))
    plt.plot((theta_bins[0:-1]+theta_bins[1:])/2, EECs, marker='o', linestyle='-', color='b', label='Weighted Avg EEC')
    plt.xlabel('cos(theta)', fontsize=14)
    plt.ylabel('Average EEC', fontsize=14)
    plt.title('Energy-Energy Correlation (EEC) vs. cos(theta)', fontsize=16)
    plt.legend(fontsize=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.show()