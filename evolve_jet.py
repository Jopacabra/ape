
import logging
import logging.handlers
import sys
import os
import json
import timeit
from pathlib import Path
import tempfile
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import pythia8
import pyhepmc as hp
# from particle import literals as lp
from IPython.display import display

import config
import pythia
import plasma
import parton_evolution
import collision
import plotting
import event_dataset
import fragmentation
import utilities


############
# Settings #
############
# Visualization options
visualize = True
visualize_2D = True
visualize_3Dz = False
visualize_3Detas = False

# Event options
num_hard_events = config.mode.NUM_HARD
event_type = config.mode.EVENT_TYPE
seed = config.mode.SEED

#############################
# Logging and File Handling #
#############################
# File paths
# Set running location as current directory - whatever the pwd was when running the script
project_path = os.path.dirname(os.path.realpath(__file__))  # Gets directory the EBE.py script is located in
home_path = os.getcwd()  # Gets working directory when script was run - results directory will be placed here
results_path = os.path.join(home_path, "results")  # Absolute path of dir where results files will live
os.makedirs(results_path, exist_ok=True)  # Make results directory

# Create result folders, if necessary
Path(os.path.join(project_path, "results/hepmc/m")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(project_path, "results/hepmc/v")).mkdir(parents=True, exist_ok=True)

# Clear any existing logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create log file & configure logging to be handled into the file AND stderr
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(os.path.join(project_path, "results/log.log"))
    ],
    force=True
)

# Ignore obnoxious font-search problems
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

# Copy config file to the results directory
config_file_path = os.path.join(project_path, 'user_config.yml')
config_file_dest = os.path.join(results_path, "hepmc", 'user_config.yml')
if not os.path.exists(config_file_dest):
    logging.info(f"Copying {config_file_path} to {config_file_dest}")
    os.system(f"cp {config_file_path} {config_file_dest}")

# Create event folders, if necessary
next_event_ii = 0
if config.mode.KEEP_EVENT:
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
        event_dir = os.path.join(project_path, f"stored_events/{event_type}/event_{next_event_ii:02d}/")
        Path(event_dir).mkdir(parents=True, exist_ok=False)
else:
    temp_dir_obj = tempfile.TemporaryDirectory()
    event_dir = temp_dir_obj.name

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
    plasma_object = collision.generate_event(working_dir=event_dir, IC_type="Duke", seed=seed)

# Create a sampled realistic DukeQCD generator event with averaged initial conditions
elif event_type == "Duke_avg":
    logging.info('Generating new averaged Duke event...')

    # Run event generation using config setttings
    # Note that we need write permissions in the working directory
    plasma_object = collision.generate_event(working_dir=event_dir, IC_type="Duke_avg", seed=seed)

# Create a brick of static plasma
elif event_type == "brick":
    logging.info("Generating x-direction slab...")
    T = 0.5
    rmax = 10

    def temp_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, T, 0.0)
    def x_vel_func(t, x, y, etas):
        return 0.0
    def y_vel_func(t, x, y, etas):
        return 0.0
    def z_vel_func(t, x, y, etas):
        return 0.0

    plasma_object = plasma.functional_plasma_3_1D(temp_func=temp_func,
                                                  x_vel_func=x_vel_func,
                                                  y_vel_func=y_vel_func,
                                                  z_vel_func=z_vel_func,
                                                  name=None, resolution=10, xmax=1.5*rmax, time=1.5*rmax, tau0=0.5)

# Create a slab of flowing plasma with flow pointing in the positive x direction
elif event_type == "slab":
    logging.info("Generating x-direction slab...")
    T = 0.4
    u = 0.7
    rmax = 7.5

    def temp_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, T, 0.0)
    def x_vel_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, u, 0.0)
    def y_vel_func(t, x, y, etas):
        return 0.0
    def z_vel_func(t, x, y, etas):
        return 0.0

    plasma_object = plasma.functional_plasma_3_1D(temp_func=temp_func,
                                                  x_vel_func=x_vel_func,
                                                  y_vel_func=y_vel_func,
                                                  z_vel_func=z_vel_func,
                                                  name=None, resolution=10, xmax=1.5*rmax, time=1.5*rmax, tau0=0.5)

# Create a slab of flowing plasma with flow and flow gradient in positive x dir.
elif event_type == "flowgradslab":
    logging.info("Generating x-direction slab...")
    T = 0.4
    u = 0.7
    gradu = 0.05
    rmax = 7.5

    def temp_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, T, 0.0)
    def x_vel_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, u + gradu*x, 0.0)
    def y_vel_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, gradu*x, 0.0)
    def z_vel_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, gradu*x, 0.0)

    plasma_object = plasma.functional_plasma_3_1D(temp_func=temp_func,
                                                  x_vel_func=x_vel_func,
                                                  y_vel_func=y_vel_func,
                                                  z_vel_func=z_vel_func,
                                                  name=None, resolution=10, xmax=1.5*rmax, time=1.5*rmax, tau0=0.5)

# Create a slab of flowing plasma with flow and T grad pointing in pos. x dir
elif event_type == "tempgradslab":
    logging.info("Generating x-direction slab...")
    T = 0.4
    gradT = 0.05
    u = 0.7
    rmax = 7.5

    def temp_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, T + gradT*x, 0.0)
    def x_vel_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, u, 0.0)
    def y_vel_func(t, x, y, etas):
        return 0.0
    def z_vel_func(t, x, y, etas):
        return 0.0

    plasma_object = plasma.functional_plasma_3_1D(temp_func=temp_func,
                                                  x_vel_func=x_vel_func,
                                                  y_vel_func=y_vel_func,
                                                  z_vel_func=z_vel_func,
                                                  name=None, resolution=10, xmax=1.5*rmax, time=1.5*rmax, tau0=0.5)

# Create a slab of flowing plasma with flow, flow grad, and T grad pointing in pos. x dir
elif event_type == "fullgradslab":
    logging.info("Generating x-direction slab...")
    T = 0.25
    gradT = 0.05
    u = 0.9
    gradu = -0.05
    rmax = 7.5

    def temp_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, T + gradT*x, 0.0)
    def x_vel_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, u + gradu * x, 0.0)
    def y_vel_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, gradu * x, 0.0)
    def z_vel_func(t, x, y, etas):
        return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, gradu * x, 0.0)

    plasma_object = plasma.functional_plasma_3_1D(temp_func=temp_func,
                                                  x_vel_func=x_vel_func,
                                                  y_vel_func=y_vel_func,
                                                  z_vel_func=z_vel_func,
                                                  name=None, resolution=10, xmax=1.5*rmax, time=1.5*rmax, tau0=0.5)

# Load a saved Duke event
else:
    try:
        # Load hydro data
        logging.info(f"Loading Plasma from {event_type}...")

        # Load event metadata
        try:
            with open(os.path.join(Path(event_type).parent, "observables.json"), "r") as f:  # Directory of .dat file
                soft_dict = json.load(f)
        except FileNotFoundError:
            logging.error("No metadata found for this event.")
            soft_dict = {}

        plasma_object = plasma.plasma(hydro_file_path=event_type)
    except:
        logging.error("Invalid event type or path.")
        raise ValueError("Invalid event type.")

logging.info('Soft event created.')


########################
# Hard Event Evolution #
########################
# Global variables that will be filled on a per-worker basis
_rng = None
_nn = None
_log_queue = None

# Particle evolution worker initializer
def _worker_init(child_seeds, worker_counter, worker_counter_lock, log_queue):
    global _rng, _nn, _log_queue
    _log_queue = log_queue

    # Remove any existing handlers and replace with QueueHandler
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)

    # Assign a guaranteed-unique worker index using a shared atomic counter
    with worker_counter_lock:
        worker_id = worker_counter.value
        worker_counter.value += 1

    # Get an RNG for this worker
    # worker_id = os.getpid() % len(child_seeds)
    _rng = np.random.default_rng(child_seeds[worker_id % len(child_seeds)])

    # Set a specific cache directory for this worker
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_worker_{worker_id}"

    # Load neural network, if in aniso_NN mode.
    if config.jet.RAD_MODEL == "aniso_NN":
        # Get the path of this file and import the radiation NN path
        script_dir = str(Path(__file__).resolve().parent)
        flow_rad_nn_dir = os.path.join(script_dir, 'flow-rad-nn/')
        sys.path.append(flow_rad_nn_dir)
        from train_radiation_nn import RadiationEmulatorInference

        # Load radiation neural network -- O(0.01s)
        logging.debug("Loading radiation neural network...")
        _nn = RadiationEmulatorInference(
            model_file=os.path.join(flow_rad_nn_dir, "data/radiation_emulator.pt"),
            normalization_file=os.path.join(flow_rad_nn_dir, "data/radiation_normalization.json"),
            device='cpu',
            compile=False,
            quiet=True,
        )

        logging.debug("Network loaded.")
    else:
        _nn = None

def should_evolve(particle):
    """Mirror of the skip conditions in treat_particle, runs in the main process."""
    if particle.status < 0:
        return False
    if np.abs(particle.rap) > config.jet.RAP_MAX_EVOLVE:
        return False
    if not particle.isg and not particle.isq and not particle.isEWB:
        return False
    return True

def treat_particle(particle, medium):
    """
    Evolve particle, then return modified particle and lists of emission momenta and emission coordinates
    """
    global _rng, _nn, _log_queue
    logging.debug('Particle {}...'.format(particle.printout()))

    #########################
    # Perform the evolution #
    #########################
    pT0 = particle.pT
    emission_momenta, emission_coords, evolution_complete = parton_evolution.evolve_particle(particle, medium, _nn, rng=_rng)
    pTF = particle.pT
    logging.debug(f"pT0: {pT0}, delta pT: {pTF - pT0} GeV")

    # Drain this worker's log records and return them
    records = []
    while not _log_queue.empty():
        records.append(_log_queue.get_nowait())

    return particle, emission_momenta, emission_coords, records

num_jets = 0  # Counter for total jets analyzed
try:

    hard_event_records = np.array([])
    for i in range(num_hard_events):
        random_label = int(utilities.rng.uniform(1000000000, 9999999999, 1)[0])
        logging.info("=" * 70)
        logging.info(
            f"Starting new hard scattering event {i + 1} of {num_hard_events} with label {random_label}."
        )
        logging.info("=" * 70)
        ##################
        # Jet Production #
        ##################

        """
        Get a hard particle event from Pythia.
        """
        # Production point
        if config.mode.VARY_POINT:
            logging.debug('Sampling hard scattering point...')
            point = collision.generate_jet_seed_point(plasma_object, seed=seed+i)
            tau_0 = config.jet.TAU_PROD
            x_0 = point[0]
            y_0 = point[1]
            etas_0 = 0.0

        else:
            logging.debug('Using central hard scattering point...')
            tau_0 = config.jet.TAU_PROD
            x_0 = 0
            y_0 = 0
            etas_0 = 0.0

        logging.info(f"Embedding hard scattering at ({tau_0}, {x_0}, {y_0}, {etas_0})")
        hard_event, event_weight = pythia.scattering(tau=tau_0, x=x_0, y=y_0, etas=etas_0, pythia_event=False, seed=seed + i)
        num_hard_particles = len(hard_event.particles)
        logging.info('Hard scattering done.')

        logging.info('Hadronizing vacuum result...')
        vacuum_event_hadrons = pythia.ape_to_pythia(hard_event)
        logging.info('Vacuum hadronization complete.')

        # Create a "live" copy of every status > 0 particle in the event that will be modified by Ape.
        hard_event.spawn_child_particles()


        #################
        # Jet Evolution #
        #################
        """
        Process each particle in the jet in parallel, then process each emission in parallel, & so on.
        
        Only the direct descendents of a prompt hard particle are evolved -- thus we do not treat higher order in 
        opacity radiation. They lose energy elastically and radiatively, but we drop the subsequently emitted particles.
        """
        round_no = 0
        passed_particles = 0
        max_rad_gens = 1  # Maximum number of emissions from a single hard particle lineage

        # Try to get max workers of Slurm environment variable, else use number of cores read by os.cpu
        if config.mode.MAX_WORKERS > 0:
            max_workers = config.mode.MAX_WORKERS
        else:
            max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        seed_sequence = np.random.SeedSequence()
        child_seeds = seed_sequence.spawn(max_workers)  # One per worker

        # Create a shared atomic counter so each worker gets a guaranteed-unique index
        worker_counter = multiprocessing.Value('i', 0)
        worker_counter_lock = multiprocessing.Lock()

        log_queue = multiprocessing.Queue()
        while True:  # Keep going until all particles are evolved
            """
            Perform the evolution on each particle
            """
            logging.info(f'Evolving particles, round {round_no}...')
            round_particles = hard_event.particles[passed_particles:]
            with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_worker_init,
                    initargs=(child_seeds, worker_counter, worker_counter_lock, log_queue)
            ) as executor:

                # Process particles in parallel
                futures = {}
                for p in round_particles:
                    if should_evolve(p):
                        futures[executor.submit(treat_particle, p, plasma_object)] = p
                    else:
                        pass

                # As they complete, spawn appropriate child particles
                for future in as_completed(futures):
                    # Get result of this process
                    modified_particle, emission_momenta, emission_coords, records = future.result()

                    # Replay log records into the main process logger
                    main_logger = logging.getLogger()
                    for record in records:
                        main_logger.handle(record)

                    # Overwrite particle with modified particle
                    hard_event.particles[modified_particle.tag] = modified_particle

                    # Spawn child particles
                    if round_no < max_rad_gens:  # Only create new particles for the first round of emissions
                        for i in range(0, len(emission_momenta)):
                            hard_event.spawn_radiation(modified_particle.tag, emission_momenta[i], emission_coords[i])
                    else:
                        pass
            passed_particles += len(round_particles)

            logging.info(f'Evolution round {round_no} complete.')
            if passed_particles == len(hard_event.particles):
                logging.info('All particles evolved.')
                break
            round_no += 1


        #################
        # Hadronization #
        #################

        """
        Hadronize hard particles using Lund-String hadronization.
        """
        if config.jet.hadronization.STRING:
            AA_pythia_event = pythia.ape_to_pythia(hard_event)  # Adds shower history
        else:
            AA_pythia_event = pythia.ape_to_pythia(hard_event, hadronize=False)  # Adds shower history
        """
        Hadronize particles using fragmentation
        """
        if config.jet.hadronization.FRAG:
            logging.info('Fragmenting hard particles...')
            fragger = fragmentation.Fragger(seed=config.mode.SEED)
            for particle in hard_event.particles:
                particle.fragz = fragger.frag(particle)
                particle.fragz0 = fragger.frag(particle, i=True)
            logging.info('Fragmentation of hard particles complete.')
        else:
            logging.info('Skipping fragmentation of hard particles.')

        ######################
        # HepMC Event output #
        ######################
        """
        Send the output of the Pythia events to a HepMC3 file.
        """
        if config.mode.WRITE_HEPMC:
            logging.debug("Saving Medium HepMC3 file...")
            hepmc_event = pythia.pythia_to_hepmc(AA_pythia_event, vt=tau_0 * np.cosh(etas_0), vx=x_0, vy=y_0, vz=tau_0 * np.sinh(etas_0), weight=event_weight)
            hepmc_filename = f"results/hepmc/m/{random_label}.dat"
            # os.remove(hepmc_filename)
            with hp.open(hepmc_filename, "w") as f:
                f.write(hepmc_event)
            logging.debug("Saved Medium HepMC3 file.")

            logging.debug("Saving Vacuum HepMC3 file...")
            vac_hepmc_event = pythia.pythia_to_hepmc(vacuum_event_hadrons, vt=tau_0*np.cosh(etas_0), vx=x_0, vy=y_0, vz=tau_0*np.sinh(etas_0), weight=event_weight)
            vac_hepmc_filename = f"results/hepmc/v/vac_{random_label}.dat"
            # os.remove(hepmc_filename)
            with hp.open(vac_hepmc_filename, "w") as f:
                f.write(vac_hepmc_event)
            logging.debug("Saved Vacuum HepMC3 file.")


        ####################################
        # Hard Particle Dataset Management #
        ####################################
        # Initialize dataset manager for saving particle data
        if config.mode.WRITE_DATAFRAME:
            logging.debug("Writing dataframe to hierarchical dataset...")
            dataset_manager = event_dataset.HierarchicalEventDataset(os.path.join(results_path, "particle_dataset"))
            job_id = int(os.environ.get("CONDOR_CLUSTER_ID", "0"))  # Extract from HTC job ID

            # Get soft event property dictionary, if present
            if plasma_object.meta is not None:
                soft_dict = plasma_object.meta
            else:
                soft_dict = {}

            # Create a config dictionary
            flat_config = {}
            for name, cls in inspect.getmembers(config, inspect.isclass):
                flat_config.update(utilities.config_to_dict(cls, prefix=name))

            # Save particles to hierarchical dataset
            dataset_manager.save_job_output(
                job_id=job_id,
                hard_id=random_label,
                soft_event_seed=seed,
                event_record=hard_event,
                soft_event_props= soft_dict,
                config_dict=flat_config,
            )

            logging.info("Particle dataset saved successfully")

        ##########################
        # Optional visualization #
        ##########################
        if visualize:
            logging.info('Visualizing...')

            # plotting.plot_trajectories(hard_event, z_axis=None, rap_max=1)
            if visualize_2D:
                plotting.plot_parton_hadron(hard_event=hard_event, hadrons=AA_pythia_event, rap_max=1.5)
            if visualize_3Dz:
                plotting.plot_trajectories(hard_event, z_axis="z", rap_max=None)
            if visualize_3Detas:
                plotting.plot_trajectories(hard_event, z_axis="etas", rap_max=None)

    try:
        logging.debug("Cleaning up temporary event directory...")
        temp_dir_obj.cleanup()
        logging.debug("Cleanup completed.")
    except NameError:
        # No temp directory
        pass

    logging.info("All events complete. Have a nice day! :)")

except KeyboardInterrupt:
    logging.info("Keyboard interrupt. Have a nice day! :)")
    pass
except Exception as e:
    logging.error("ACK! Something went wrong.")
    logging.exception(e)

