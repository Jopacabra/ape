
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
import plotting


############
# Settings #
############
# Analysis options
analyze = True
theta_bins = np.linspace(-1, 1, 21)
EECs = np.zeros(len(theta_bins)-1)

# Visualization options
visualize = True

# Event options
num_hard_events = 1
event_type = "Duke"
plasma_file_path = "stored_events/Duke_avg/event_0/viscous_14_moments_evo.dat"


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
logging.getLogger('matplotlib.ticker').disabled = True


########################
# Soft Event Evolution #
########################

"""
Select an event type and generate a plasma object.
"""
# Create a ws optical glauber plasma.
if event_type == "ws":
    logging.info('Creating ws optical glauber plasma...')
    plasma_object = collision.woods_saxon_plasma(b=5.5, T0=0.39, A=208, a=0.546, alpha=1, name=None,
                           resolution=5, rmax=10, tmin=0.5, tmax=None, umax=1, return_grids=False)

# Create a gaussian optical glauber plasma.
elif event_type == "gaussian":
    logging.info('Creating gaussian optical glauber plasma...')
    plasma_object = collision.gaussian_plasma(b=5, T0=0.39, A=208, alpha=0.2, name=None,
                           resolution=5, rmax=10, tmin=0.5, tmax=None, umax=0.75, return_grids=False)

# Create a sampled realistic DukeQCD generator event.
elif event_type == "Duke":
    logging.info('Generating new event...')

    # Run event generation using config setttings
    # Note that we need write permissions in the working directory
    plasma_object = collision.generate_event(working_dir=None, IC_type="Duke")

# Load a saved Duke event
elif event_type == "load":
    logging.info("Loading Duke Average Plasma...")
    plasma_file = plasma.osu_hydro_file(plasma_file_path)
    plasma_object = plasma.plasma_event(hydro_object=plasma_file)

else:
    logging.error("Invalid event type.")
    raise ValueError("Invalid event type.")

logging.info('Plasma created.')


########################
# Hard Event Evolution #
########################
num_jets = 0  # Counter for total jets analyzed
try:

    hard_event_records = np.array([])
    for i in range(num_hard_events):
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

            # plotting.plot_trajectories(hard_event, z_axis=None, rap_max=1)
            plotting.plot_parton_hadron(hard_event=hard_event, hadrons=hard_event_hadrons, rap_max=1.5)
            # plotting.plot_trajectories(hard_event, z_axis="z", rap_max=5)
            # hard_event.plot_trajectories(z_axis="etas", rap_max=5)
            # plt.savefig("particle_trajectories.png", dpi=150)

        #####################
        # Optional analysis #
        #####################
        if analyze:
            logging.info('Analyzing...')

            # Find jets
            jets = pythia.pythia_to_fastjet(pythia_had=hard_event_hadrons, rap_max=1.5, R=0.4, pTmin=0.0)

            # Cut to jets we care about
            analyzed_jets = []
            for jet in jets:
                if jet.pt() > 10:
                    analyzed_jets.append(jet)

            # Compute EECs for jets
            for jet in analyzed_jets:
                current_EECs, _ = observables.EEC(jet, plot=False, bins=theta_bins)
                EECs = EECs + current_EECs
                num_jets += 1

except KeyboardInterrupt:
    pass
except Exception as e:
    logging.exception(e)

###########################
# Post Evolution Analysis #
###########################


if analyze:
    # Save result
    np.savez("EECs.npz", EEC_sum=EECs, theta_bins=theta_bins, num_jets=np.array([num_jets]))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot((theta_bins[0:-1]+theta_bins[1:]) / 2, EECs / num_jets, marker='o', linestyle='-', color='b', label='Weighted Avg EEC')
    plt.xlabel('cos(theta)', fontsize=14)
    plt.ylabel('Average EEC', fontsize=14)
    plt.title('Energy-Energy Correlation (EEC) vs. cos(theta)', fontsize=16)
    plt.legend(fontsize=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

