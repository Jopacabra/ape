
import logging
import sys
import os
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pythia8
import pyhepmc as hp
# from particle import literals as lp
from IPython.display import display



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
# Visualization options
visualize = False

# Event options
num_hard_events = 3334
event_type = "load"
plasma_file_path = "stored_events/Duke_avg/event_01/viscous_14_moments_evo.dat"


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

# Create result folders, if necessary
Path("results/hepmc/m").mkdir(parents=True, exist_ok=True)
Path("results/hepmc/v").mkdir(parents=True, exist_ok=True)

# Create event folders, if necessary
if event_type == "Duke_avg" or event_type == "Duke":
    # Find which event to save as
    Path(f"stored_events/{event_type}").mkdir(parents=True, exist_ok=True)


    # Find the next event number
    dir_path = Path(f"stored_events/{event_type}")

    # Get all subdirectories matching the pattern event_xx
    used_numbers = set()

    # Iterate and collect used numbers
    for subdir in dir_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith("event_"):
            try:
                # Extract the number part after "event_"
                num_str = subdir.name[6:]  # Skip "event_"
                num = int(num_str)
                if 0 <= num <= 99:  # Only consider valid two-digit numbers
                    used_numbers.add(num)
            except ValueError:
                # Skip directories that don't match the expected format
                pass

    # Find the lowest unused number
    for i in range(100):
        if i not in used_numbers:
            next_event_ii = i
            break

    logging.info(f"Event saving as event_{next_event_ii:02d}.")

    # Create event directory
    event_dir = f"stored_events/{event_type}/event_{next_event_ii:02d}/"
    Path(event_dir).mkdir(parents=True, exist_ok=False)


########################
# Soft Event Evolution #
########################

"""
Select an event type and generate a plasma object.
"""
# Create a ws optical glauber plasma.
if event_type == "ws":
    logging.info('Creating ws optical glauber plasma...')
    plasma_object = collision.woods_saxon_plasma(b=5.5, T0=0.30, A=208, a=0.546, alpha=1, name=None,
                           resolution=5, rmax=10, tmin=0.5, tmax=None, umax=1, return_grids=False)

# Create a gaussian optical glauber plasma.
elif event_type == "gaussian":
    logging.info('Creating gaussian optical glauber plasma...')
    plasma_object = collision.gaussian_plasma(b=5, T0=0.39, A=208, alpha=0.2, name=None,
                           resolution=5, rmax=10, tmin=0.5, tmax=None, umax=0.75, return_grids=False)

# Create a sampled realistic DukeQCD generator event.
elif event_type == "Duke":
    logging.info('Generating new Duke event...')

    # Run event generation using config setttings
    # Note that we need write permissions in the working directory
    plasma_object = collision.generate_event(working_dir=results_path, IC_type="Duke")

# Create a sampled realistic DukeQCD generator event with averaged initial conditions
elif event_type == "Duke_avg":
    logging.info('Generating new averaged Duke event...')

    # Run event generation using config setttings
    # Note that we need write permissions in the working directory
    plasma_object = collision.generate_event(working_dir=results_path, IC_type="Duke_avg")

# Load a saved Duke event
elif event_type == "load":
    logging.info(f"Loading Plasma from {plasma_file_path}...")
    plasma_file = plasma.osu_hydro_file(plasma_file_path)
    plasma_object = plasma.plasma_event(hydro_object=plasma_file)

else:
    logging.error("Invalid event type.")
    raise ValueError("Invalid event type.")

logging.info('Soft event complete created.')


########################
# Hard Event Evolution #
########################
num_jets = 0  # Counter for total jets analyzed
try:

    hard_event_records = np.array([])
    for i in range(num_hard_events):
        rng = np.random.default_rng()
        random_label = int(rng.uniform(1000000000, 9999999999, 1)[0])
        logging.info(
            f"Starting new hard scattering event {i + 1} of {num_hard_events} with label {random_label}."
        )
        ##################
        # Jet Production #
        ##################

        """
        Get a hard particle event from Pythia.
        """
        # Production point
        if config.mode.VARY_POINT:
            logging.debug('Sampling hard scattering point...')
            point = collision.generate_jet_seed_point(plasma_object)
            tau_0 = config.jet.TAU_PROD
            x_0 = point[0]
            y_0 = point[1]
            etas_0 = 0.0

        else:
            logging.debug('Using central hard scattering point...')
            tau_0 = config.jet.TAU_PROD
            x_0 = 1
            y_0 = 1
            etas_0 = 0.0

        logging.info(f"Embedding hard scattering at ({tau_0}, {x_0}, {y_0}, {etas_0})")
        hard_event, event_weight, pythia_record = pythia.scattering(tau=tau_0, x=x_0, y=y_0, etas=etas_0, pythia_event=True)
        num_hard_particles = len(hard_event.particles)
        logging.info('Hard scattering done.')

        logging.info('Hadronizing vacuum result...')
        vacuum_event_hadrons = pythia.pp_shower_hadronize(hard_event, pythia_record)  # Adds shower history
        logging.info('Vacuum hadronization complete.')


        #################
        # Jet Evolution #
        #################

        """
        Perform the evolution on each particle
        """
        logging.info('Evolving particles...')
        for particle in hard_event.particles:
            pT0 = particle.pT
            parton_evolution.evolve_particle(particle, plasma_object)
            pTF = particle.pT
            logging.debug(f"Particle delta pT: {pTF - pT0} GeV")
        logging.info('Particle evolution complete')


        #################
        # Hadronization #
        #################

        """
        Hadronize hard particles using Lund-String hadronization.
        """
        hard_event_hadrons = pythia.pp_shower_hadronize(hard_event, pythia_record)  # Adds shower history
        # for p in hard_event_hadrons.particles():
        #     print(p.statusHepMC())

        ################
        # Event output #
        ################
        """
        Send the output of the Pythia events to a HepMC3 file.
        """
        logging.debug("Saving Medium HepMC3 file...")
        hepmc_event = pythia.pythia_to_hepmc(hard_event_hadrons, vt=tau_0*np.cosh(etas_0), vx=x_0, vy=y_0, vz=tau_0*np.sinh(etas_0), weight=event_weight)
        hepmc_filename = f"results/hepmc/m/{random_label}.dat"
        # os.remove(hepmc_filename)
        with hp.open(hepmc_filename, "w") as f:
            f.write(hepmc_event)

        logging.debug("Saving Vacuum HepMC3 file...")
        vac_hepmc_event = pythia.pythia_to_hepmc(vacuum_event_hadrons, vt=tau_0*np.cosh(etas_0), vx=x_0, vy=y_0, vz=tau_0*np.sinh(etas_0), weight=event_weight)
        vac_hepmc_filename = f"results/hepmc/v/vac_{random_label}.dat"
        # os.remove(hepmc_filename)
        with hp.open(vac_hepmc_filename, "w") as f:
            f.write(vac_hepmc_event)

        # graph_filename = "hadronic_event.svg"
        # os.remove(graph_filename)
        # hp.view.savefig(hepmc_event, graph_filename)

        ##########################
        # Optional visualization #
        ##########################
        if visualize:
            logging.info('Visualizing...')

            # plotting.plot_trajectories(hard_event, z_axis=None, rap_max=1)
            plotting.plot_parton_hadron(hard_event=hard_event, hadrons=hard_event_hadrons, rap_max=1.5)
            plotting.plot_trajectories(hard_event, z_axis="z", rap_max=None)
            plotting.plot_trajectories(hard_event, z_axis="etas", rap_max=None)

    logging.info("All events complete. Have a nice day! :)")

except KeyboardInterrupt:
    logging.info("Keyboard interrupt. Have a nice day! :)")
    pass
except Exception as e:
    logging.error("ACK! Something went wrong.")
    logging.exception(e)

