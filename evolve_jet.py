
import logging
import sys
import os
import timeit

import numpy as np
import matplotlib.pyplot as plt

import config
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
costheta_bins = np.linspace(-1, 1, 21)
EECs = np.zeros(len(costheta_bins) - 1)
E_bins = np.linspace(1, 15, 5)
v1s = np.zeros(len(E_bins) - 1)
v2s = np.zeros(len(E_bins) - 1)

# Visualization options
visualize = False

# Event options
num_hard_events = 1
event_type = "Duke"
plasma_file_path = "stored_events/Duke_avg/event_0/viscous_14_moments_evo.dat"


#############################
# Logging and File Handling #
#############################
# File paths
# Set running location as current directory - whatever the pwd was when running the script
project_path = os.path.dirname(os.path.realpath(__file__))  # Gets directory the EBE.py script is located in
home_path = os.getcwd()  # Gets working directory when script was run - results directory will be placed here
results_path = home_path + "/results"  # Absolute path of dir where results files will live
os.makedirs(results_path, exist_ok=True)  # Make results directory

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
    plasma_object = collision.generate_event(working_dir=results_path, IC_type="Duke")

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
        if config.mode.VARY_POINT:
            logging.info('Sampling hard scattering point...')
            point = collision.generate_jet_seed_point(plasma_object)
            tau_0 = config.jet.TAU_PROD
            x_0 = point[0]
            y_0 = point[1]
            etas_0 = 0.0

        else:
            logging.info('Using central hard scattering point...')
            tau_0 = config.jet.TAU_PROD
            x_0 = 1
            y_0 = 1
            etas_0 = 0.0
        logging.info(f"Embedding hard scattering at ({tau_0}, {x_0}, {y_0}, {etas_0})")
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
            plotting.plot_parton_hadron(hard_event=hard_event, hadrons=hard_event_hadrons, rap_max=1.5,
                                        final_tau=plasma_object.tf)
            plotting.plot_trajectories(hard_event, z_axis="z", rap_max=None)
            plotting.plot_trajectories(hard_event, z_axis="etas", rap_max=None)

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

            # Compute observables for jets
            for jet in analyzed_jets:
                # EECs
                current_EECs, _ = observables.EEC(jet=jet, plot=False, bins=costheta_bins)
                EECs = EECs + current_EECs

                # vns
                current_v1s, _ = observables.fastjet_intrajetvnish(jet=jet, pT_min=1, alpha_0=0, E_bins=E_bins, n=1)
                v1s = v1s + (current_v1s)
                current_v2s, _ = observables.fastjet_intrajetvnish(jet=jet, pT_min=1, alpha_0=0, E_bins=E_bins, n=2)
                v2s = v2s + (current_v2s)


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
    np.savez("EECs.npz", EEC_sum=EECs, costheta_bins=costheta_bins, num_jets=np.array([num_jets]))
    np.savez("intrajet_vns_med_ref.npz", v1_sum=v1s, v2_sum=v2s, E_bins=E_bins, num_jets=np.array([num_jets]))

    # Plot
    # plt.figure(figsize=(8, 6))
    # plt.plot((costheta_bins[0:-1] + costheta_bins[1:]) / 2, EECs / num_jets, marker='o', linestyle='-', color='b', label='Weighted Avg EEC')
    # plt.xlabel('cos(theta)', fontsize=14)
    # plt.ylabel('Average EEC', fontsize=14)
    # plt.title('Energy-Energy Correlation (EEC) vs. cos(theta)', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.grid(True)
    # plt.show()

