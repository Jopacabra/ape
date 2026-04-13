import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import integrate
from scipy.ndimage import rotate
from scipy.ndimage import shift
try:
    import matplotlib.pyplot as plt
except:
    print('NO MATPLOTLIB')
import h5py
import math
import os
import json
import shutil
import logging
import config
from hic import initial
import utilities
import plasma
from itertools import groupby

try:
    import freestream
except:
    logging.warning('freestream not found.')
try:
    import frzout
except:
    logging.warning('frzout not found.')

"""
This module is responsible for all processes related to event generation & hard scattering.
- Rejection samples initial temperature profile for jet production
- Produces PDFs from temp backgrounds
- Planned inclusions: Pythia handling, etc.
"""

class StopEvent(Exception):
    """ Raise to end an event early. """

# Function that generates a new Trento collision event with parameters from config file.
# Returns the Trento output file name.
def runTrento(randomSeed=None, numEvents=1, quiet=False, output=None,
              bmin=config.transport.trento.BMIN, bmax=config.transport.trento.BMAX,
              grid_step=config.transport.GRID_STEP, grid_max=config.transport.GRID_MAX_TARGET,
              norm=config.transport.trento.NORM,
              cross_section=config.transport.trento.CROSS_SECTION,
              nucleon_width=config.transport.trento.NUCLEON_WIDTH,
              p=config.transport.trento.P,
              k=config.transport.trento.K,
              v=config.transport.trento.V,
              nc=config.transport.trento.NC,
              dmin=config.transport.trento.DMIN,
              return_process=False):

    # Make sure there's no file where we want to stick it.
    try:
        os.remove(output)
    except IsADirectoryError:
        shutil.rmtree(output)
    except FileNotFoundError:
        pass
    except TypeError:
        pass

    # Generate Trento command argument list
    trentoCmd = ['trento']
    if randomSeed is not None:
        trentoCmd.append('--random-seed {}'.format(int(randomSeed)))  # Random seed for repeatability
    trentoCmd.append(f'--number-events {numEvents}')  # Number of events to generate
    trentoCmd.append('--projectile {}'.format(config.transport.trento.PROJ1))  # Collision projectile 1
    trentoCmd.append('--projectile {}'.format(config.transport.trento.PROJ2))  # Collision projectile 2

    # File output options
    if output is not None:
        trentoCmd.append('--output {}'.format(output))  # Output file name

    # Numerical grid options
    trentoCmd.append(f'--grid-step {grid_step}')  # Grid step size in fm
    trentoCmd.append(f'--grid-max {grid_max}')  # Grid max size in fm

    # Physical Options
    if bmin is not None:
        trentoCmd.append('--b-min {}'.format(bmin))  # Minimum impact parameter (in fm)
    if bmax is not None:
        trentoCmd.append('--b-max {}'.format(bmax))  # Maximum impact parameter (in fm)
    trentoCmd.append('--normalization {}'.format(norm))
    trentoCmd.append('--cross-section {}'.format(cross_section))
    trentoCmd.append('--nucleon-width {}'.format(nucleon_width))
    trentoCmd.append('--reduced-thickness {}'.format(p))
    trentoCmd.append('--fluctuation {}'.format(k))
    trentoCmd.append('--constit-width {}'.format(v))
    trentoCmd.append('--constit-number {}'.format(nc))
    trentoCmd.append('--nucleon-min-dist {}'.format(dmin))
    trentoCmd.append('--ncoll')


    # Run Trento command
    # Note star unpacks the list to pass the command list as arguments
    if not quiet:
        logging.info('format: event_number impact_param npart ncoll mult e2_re e2_im e3_re e3_im e4_re e4_im e5_re e5_im')
    subprocess, output = utilities.run_cmd(*trentoCmd, quiet=quiet)

    # Parse shell output and pass to dataframe.
    first = True
    for line in output:
        trentoOutput = line.split()
        try:
            trentoDataFrame = pd.DataFrame(
                {
                    "event": [int(trentoOutput[0])],
                    "b": [float(trentoOutput[1])],
                    "npart": [int(trentoOutput[2])],
                    "ncoll": [float(trentoOutput[3])],
                    "mult": [float(trentoOutput[4])],
                    "e2_re": [float(trentoOutput[5])],
                    "e2_im": [float(trentoOutput[6])],
                    "psi_e2": [float((1/2) * np.arctan2(float(trentoOutput[6]), float(trentoOutput[5])))],
                    "e3_re": [float(trentoOutput[7])],
                    "e3_im": [float(trentoOutput[8])],
                    "psi_e3": [float((1/3) * np.arctan2(float(trentoOutput[8]), float(trentoOutput[7])))],
                    "e4_re": [float(trentoOutput[9])],
                    "e4_im": [float(trentoOutput[10])],
                    "psi_e4": [float((1/4) * np.arctan2(float(trentoOutput[10]), float(trentoOutput[9])))],
                    "e5_re": [float(trentoOutput[11])],
                    "e5_im": [float(trentoOutput[12])],
                    "psi_e5": [float((1/5) * np.arctan2(float(trentoOutput[12]), float(trentoOutput[11])))],
                    "seed": [randomSeed]
                }
            )
        except ValueError:
            trentoDataFrame = pd.DataFrame({})
        if first:
            resultsDataFrame = trentoDataFrame
            first = False
        else:
            resultsDataFrame = pd.concat([resultsDataFrame, trentoDataFrame])

        # Compute trento ic eccentricities
        resultsDataFrame['e2'] = np.sqrt(resultsDataFrame['e2_re'] ** 2 + resultsDataFrame['e2_im'] ** 2)
        resultsDataFrame['e3'] = np.sqrt(resultsDataFrame['e3_re'] ** 2 + resultsDataFrame['e3_im'] ** 2)
        resultsDataFrame['e4'] = np.sqrt(resultsDataFrame['e4_re'] ** 2 + resultsDataFrame['e4_im'] ** 2)
        resultsDataFrame['e5'] = np.sqrt(resultsDataFrame['e5_re'] ** 2 + resultsDataFrame['e5_im'] ** 2)

    # Pass on result file name, trentoSubprocess data, and dataframe.
    if return_process:
        return resultsDataFrame, output, subprocess
    else:
        return resultsDataFrame

# Define function to generate an averaged initial condition at the impact parameter associated with the seed
def runTrento_Avg(directory, randomSeed=None, quiet=False, bmin=None, bmax=None, num_events=1000):

    # Run a single event to get an impact parameter using Trento's sampling
    logging.info('Finding impact parameter from sample event...')
    event_dataframe = runTrento(output=None, randomSeed=randomSeed,
                                numEvents=1, quiet=quiet,
                                bmin=bmin, bmax=bmax)
    chosen_b = float(event_dataframe['b'].iloc[0])

    # Run many events at the chosen impact parameter
    logging.info(f'Generating {num_events} Trento events for averaging...')
    event_dataframe = runTrento(output=directory, randomSeed=None,
                                numEvents=num_events, quiet=quiet,
                                bmin=chosen_b, bmax=chosen_b)

    # Get the mean values of number of participants and number of binary collisions
    npart = event_dataframe['npart'].mean()
    ncoll = event_dataframe["ncoll"].mean()

    logging.info('Aligning and averaging events...')
    # Load the events from file and sum them -- shift to match centers of mass, rotate to match psi2
    gridstep = config.transport.GRID_STEP
    first = True
    for file in os.listdir(directory):
        # Load file as ic object
        new = np.loadtxt(directory + '/' + file)
        ic = initial.IC(new, 0.1)

        # Find center of mass and shift to 0,0
        new_cm = ic.cm()
        new = shift(new, shift=(np.array([new_cm[1], -new_cm[0]]) / gridstep))

        # Create a new ic object with the shifted data and rotate to match psi2
        ic = initial.IC(new, 0.1)
        e2_more, e2_psi2 = utilities.ecc_more(ic, 2)
        new = rotate(new, angle=-(e2_psi2 / np.pi) * 180, reshape=False)

        # Add to the summing ic array
        if first:
            ic_array = new
            first = False
        else:
            ic_array = ic_array + new

    # Normalize sum to average
    ic_array = ic_array / num_events

    logging.info('Computing event info and packaging...')
    ic_object = initial.IC(ic_array, config.transport.GRID_STEP)

    ic_mult = ic_object.sum()
    e2, psi_e2 = utilities.ecc_more(ic_object, 2)
    e3, psi_e3 = utilities.ecc_more(ic_object, 3)
    e4, psi_e4 = utilities.ecc_more(ic_object, 4)
    e5, psi_e5 = utilities.ecc_more(ic_object, 5)


    trentoDataFrame = pd.DataFrame(
        {
            "b": [float(chosen_b)],
            "mult": [float(ic_mult)],
            "npart": [int(npart)],
            "ncoll": [int(ncoll)],
            "e2": [float(e2)],
            "psi_e2": [float(psi_e2)],
            "e3": [float(e3)],
            "psi_e3": [float(psi_e3)],
            "e4": [float(e4)],
            "psi_e4": [float(psi_e4)],
            "e5": [float(e5)],
            "psi_e5": [float(psi_e5)],
            "seed": [randomSeed]
        }
    )

    return ic_array, trentoDataFrame


# Define function to generate initial conditions object as for freestream input from trento file
def toFsIc(initial_file='initial.hdf', quiet=False):
    if not quiet:
        logging.info('Packaging initial conditions array for: {}'.format(initial_file))

    with h5py.File(initial_file, 'r') as f:
        for dset in f.values():
            logging.info(dset)
            ic = np.array(dset)
    return ic

# Function adapted from DukeQCD to run osu-hydro from the freestreamed initial conditions yielded by freestream
# Result files SHOULD be placed in the active folder.
def run_hydro(fs, event_size, grid_step=0.1, tau_fs=0.5, eswitch=0.110, coarse: bool | float = False, hydro_args=None, quiet=False,
              time_step=0.1, maxTime=None):
    """
    The handling of osu-hydro implemented here is adapted directly from DukeQCD's hic-eventgen package.
    https://github.com/Duke-QCD/hic-eventgen
    ---

    Run the initial condition contained in FreeStreamer object `fs` through
    osu-hydro on a grid with approximate physical size `event_size` [fm].
    Return a dict of freeze-out surface data suitable for passing directly
    to frzout.Surface.

    Initial condition arrays are cropped or padded as necessary.

    If `coarse` is an integer > 1, use only every `coarse`th cell from the
    initial condition arrays (thus increasing the physical grid step size
    by a factor of `coarse`).  Ignore the user input `hydro_args` and
    instead run ideal hydro down to a low temperature.

    `dt_ratio` sets the timestep as a fraction of the spatial step
    (dt = dt_ratio * dxy).  The SHASTA algorithm requires dt_ratio < 1/2.

    """
    dxy = grid_step * (coarse or 1)
    ls = math.ceil(event_size/dxy)  # the osu-hydro "ls" parameter
    n = 2*ls + 1  # actual number of grid cells

    for fmt, f, arglist in [
            ('ed', fs.energy_density, [()]),
            ('u{}', fs.flow_velocity, [(1,), (2,)]),
            ('pi{}{}', fs.shear_tensor, [(1, 1), (1, 2), (2, 2)]),
    ]:
        for a in arglist:
            X = f(*a)

            if coarse:
                X = X[::coarse, ::coarse]

            diff = X.shape[0] - n
            start = int(abs(diff)/2)

            if diff > 0:
                # original grid is larger -> cut out middle square
                s = slice(start, start + n)
                X = X[s, s]
            elif diff < 0:
                # original grid is smaller
                #  -> create new array and place original grid in middle
                Xn = np.zeros((n, n))
                s = slice(start, start + X.shape[0])
                Xn[s, s] = X
                X = Xn

            X.tofile(fmt.format(*a) + '.dat')

    dt = time_step

    if coarse:
        hydroCmd = ['osu-hydro', 't0={} dt={} dxy={} nls={} edec={}'.format(tau_fs, dt, dxy, ls, eswitch),
                    'etas_hrg=0 etas_min=0 etas_slope=0 zetas_max=0 zetas_width=0']
    else:
        if maxTime is not None:
            logging.info('Limiting time...')
            hydroCmd = ['osu-hydro', 't0={} dt={} dxy={} nls={} vismin={} visslope={} viscrv={} visbulkmax={} '.format(
                                                                                tau_fs, dt, dxy, ls,
                                                                                config.transport.hydro.ETAS_MIN,
                                                                                config.transport.hydro.ETAS_SLOPE,
                                                                                config.transport.hydro.ETAS_CURV,
                                                                                config.transport.hydro.ZETAS_MAX)
                        + 'visbulkwidth={} visbulkt0={} time_stepmaxt={} edec={}'.format(config.transport.hydro.ZETAS_WIDTH,
                                                                                 config.transport.hydro.ZETAS_T0,
                                                                                 maxTime, eswitch)]
        else:
            hydroCmd = ['osu-hydro', 't0={} dt={} dxy={} nls={} vismin={} visslope={} viscrv={} visbulkmax={} '.format(
                                                                                tau_fs, dt, dxy, ls,
                                                                                config.transport.hydro.ETAS_MIN,
                                                                                config.transport.hydro.ETAS_SLOPE,
                                                                                config.transport.hydro.ETAS_CURV,
                                                                                config.transport.hydro.ZETAS_MAX)
                        + 'visbulkwidth={} visbulkt0={} edec={}'.format(config.transport.hydro.ZETAS_WIDTH,
                                                                config.transport.hydro.ZETAS_T0, eswitch)]

        if hydro_args != None:
            hydroCmd = hydroCmd + hydro_args

    hydroProc, hydroOutput = utilities.run_cmd(*hydroCmd, quiet=False)

    if not quiet:
        logging.info('format: ITime, Time, Max Energy Density, Max Temp, iRegulateCounter, iRegulateCounterBulkPi')

    surface = np.fromfile('surface.dat', dtype='f8').reshape(-1, 16)

    # end event if the surface is empty -- this occurs in ultra-peripheral
    # events where the initial condition doesn't exceed Tswitch
    if surface.size == 0:
        raise StopEvent('empty surface')

    # surface columns:
    #   0    1  2  3         4         5         6    7
    #   tau  x  y  dsigma_t  dsigma_x  dsigma_y  v_x  v_y
    #   8     9     10    11    12    13    14    15
    #   pitt  pitx  pity  pixx  pixy  piyy  pizz  Pi

    # pack surface data into a dict suitable for passing to frzout.Surface
    return dict(
        zip(['x', 'sigma', 'v'], np.hsplit(surface, [3, 6, 8])),
        pi=dict(zip(['xx', 'xy', 'yy'], surface.T[11:14])),
        Pi=surface.T[15]
    )


# Function to generate a new HIC event and dump the files in the current working directory.
def generate_event(grid_max_target=config.transport.GRID_MAX_TARGET, grid_step=config.transport.GRID_STEP,
                   time_step=config.transport.TIME_STEP, tau_fs=config.transport.hydro.TAU_FS,
                   t_end=config.transport.hydro.T_SWITCH, seed=None, working_dir=None,
                   IC_type='Duke', bmin=None, bmax=None):

    if working_dir is not None:
        og_dir = os.getcwd()
        os.chdir(working_dir)

    # the "target" grid max: the grid shall be at least as large as the target
    # By defualt grid_max_target = config.transport.GRID_MAX_TARGET
    # next two lines set the number of grid cells and actual grid max,
    # which will be >= the target (same algorithm as trento)
    grid_n = math.ceil(2 * grid_max_target / grid_step)
    grid_max = .5 * grid_n * grid_step
    logging.info(
        'grid step = %.6f fm, n = %d, max = %.6f fm',
        grid_step, grid_n, grid_max
    )

    ############################
    # Dump DukeQCD definitions #
    ############################
    # species (name, ID) for identified particle observables
    species = [
        ('pion', 211),
        ('kaon', 321),
        ('proton', 2212),
        ('Lambda', 3122),
        ('Sigma0', 3212),
        ('Xi', 3312),
        ('Omega', 3334),
    ]

    # fully specify numeric data types, including endianness and size, to
    # ensure consistency across all machines
    float_t = '<f8'
    int_t = '<i8'
    complex_t = '<c16'

    # results dictionary
    results = {}

    # UrQMD raw particle format
    parts_dtype = [
        ('sample', int),
        ('ID', int),
        ('charge', int),
        ('pT', float),
        ('ET', float),
        ('mT', float),
        ('phi', float),
        ('y', float),
        ('eta', float)
    ]

    ###############################
    # Trento / Initial Conditions #
    ###############################

    # Choose random seed
    if seed is None:
        seed = int(np.random.uniform(0, 10000000000000000))
    logging.info('Random seed selected: {}'.format(seed))

    # Default to duke events
    if IC_type == 'None':
        IC_type = 'Duke_avg'

    if IC_type == 'Duke':
        # Decide where to locate the initial conditions file
        if working_dir is not None:
            trento_ic_path = os.path.join(working_dir, '/initial.hdf')
        else:
            trento_ic_path = 'initial.hdf'

        # Debug pwd
        logging.info('Running trento in...')
        logging.info(os.getcwd())

        # Generate trento event
        event_dataframe = runTrento(randomSeed=seed,
                                    quiet=False,
                                    output=trento_ic_path,
                                    bmin=bmin, bmax=bmax)

        # Debug pwd
        logging.info('Running freestream in...')
        logging.info(os.getcwd())

        # Format trento data into initial conditions for freestream
        logging.info('Packaging trento initial conditions into array...')
        ic = toFsIc(initial_file=trento_ic_path, quiet=False)
    elif IC_type == 'Duke_avg':
        # Decide where to locate the initial conditions files
        if working_dir is not None:
            trento_ic_path = os.path.join(working_dir, 'trento_output')
        else:
            trento_ic_path = 'trento_output'

        # Get an averaged initial condition from many trento runs
        ic_array, event_dataframe = runTrento_Avg(directory=trento_ic_path, randomSeed=seed,
                                        quiet=False, bmin=bmin, bmax=bmax)

        ic = ic_array

    # Form a flat dictionary of the event results
    results["b"] = float(event_dataframe['b'].iloc[0])
    results["mult"] = float(event_dataframe['mult'].iloc[0])
    results["npart"] = int(event_dataframe['npart'].iloc[0])
    results["ncoll"] = int(event_dataframe['ncoll'].iloc[0])
    results["e2"] = float(event_dataframe['e2'].iloc[0])
    results["psi_e2"] = float(event_dataframe['psi_e2'].iloc[0])
    results["e3"] = float(event_dataframe['e3'].iloc[0])
    results["psi_e3"] = float(event_dataframe['psi_e3'].iloc[0])
    results["e4"] = float(event_dataframe['e4'].iloc[0])
    results["psi_e4"] = float(event_dataframe['psi_e4'].iloc[0])
    results["e5"] = float(event_dataframe['e5'].iloc[0])
    results["psi_e5"] = float(event_dataframe['psi_e5'].iloc[0])
    results["seed"] = seed

    #################
    # Freestreaming #
    #################
    # Freestream initial conditions
    logging.info('Freestreaming Trento conditions...')
    fs = freestream.FreeStreamer(initial=ic, grid_max=grid_max, time=tau_fs)

    # Compute initial entropy
    results['initial_entropy'] = ic.sum() * grid_step ** 2

    # Important to close the hdf5 file.
    del ic

    #########
    # Hydro #
    #########
    # Run hydro on initial conditions
    # This is where we control the end point of the hydro. The HRG object created here has an energy density param.
    # that we use as the cut-off energy density for the hydro evolution. Doing things through frzout.HRG allows us to
    # specify a minimum temperature that will be enforced with the energy density popped out here.
    # create frzout HRG object (to be reused for all events) representing a hadron resonance gas at given temperature
    hrg_kwargs = dict(species='urqmd', res_width=True)
    hrg = frzout.HRG(t_end, **hrg_kwargs)
    hrg_coarse = frzout.HRG(0.110, **hrg_kwargs)

    # append switching energy density to hydro arguments
    # We use frzout's hrg class to compute an energy density based on the desired freezeout temperature
    eswitch = hrg.energy_density()
    eswitch_coarse = hrg_coarse.energy_density()


    # Coarse run to determine maximum radius
    logging.info('Running coarse hydro...')
    coarseHydroDict = run_hydro(fs, event_size=27, coarse=3, grid_step=grid_step,
                                tau_fs=tau_fs, eswitch=eswitch_coarse,
                                time_step=time_step)
    rmax = math.sqrt((
                             coarseHydroDict['x'][:, 1:3] ** 2
                     ).sum(axis=1).max())
    logging.info('rmax = %.3f fm', rmax)

    # Determine maximum number of timesteps needed
    # This is the time it takes for a jet to travel across the plasma on its longest path at the speed of light
    # Note that this isn't the longest pathlength in the grid, it's the longest pathlength in the fluid at final time
    # Right now, we don't actually use this, but it's on the docket to consider whether the edge cases that see the
    # end of event time data are a statistically significant fraction of events.
    # maxTime = 2*rmax  # in fm --- equal to length to traverse in fm for c = 1 - 2x largest width of plasma
    # logging.info('maxTime = %.3f fm', maxTime)

    # Dump the coarse run event data
    logging.info('Dumping coarse run hydro data')
    utilities.run_cmd(*['rm', 'viscous_14_moments_evo.dat'],
                      quiet=False)


    # Fine run
    logging.info('Running fine hydro...')
    hydro_dict = run_hydro(fs, event_size=rmax, grid_step=grid_step, tau_fs=tau_fs,
              eswitch=eswitch, time_step=time_step)

    ##########
    # Frzout #
    ##########

    # Compute flow coefficients v_n:
    # Create event surface object from hydro surface file dictionary
    event_surface = frzout.Surface(**hydro_dict, ymax=2)
    logging.info('%d freeze-out cells', len(event_surface))

    minsamples, maxsamples = 10, 1000  # reasonable range for nsamples
    minparts = 10 ** 5  # min number of particles to sample
    nparts = 0  # for tracking total number of sampled particles

    logging.info('sampling surface with frzout')

    # sample particles and write to file
    with open(os.path.join(working_dir, 'particles_in.dat'), 'w') as f:
        for nsamples in range(1, maxsamples + 1):
            parts = frzout.sample(event_surface, hrg)
            if parts.size == 0:
                continue
            nparts += parts.size
            print('#', parts.size, file=f)
            for p in parts:
                print(p['ID'], *p['x'], *p['p'], file=f)
            if nparts >= minparts and nsamples >= minsamples:
                break

    logging.info('produced %d particles in %d samples', nparts, nsamples)
    results['n_cf_samples'] = nsamples  # Number of Cooper-Fry freezeout samples

    if nparts == 0:
        raise StopEvent('no particles produced')

    ###################################
    # Log event size and eccentricity #
    ###################################

    # # try to free some memory
    # # (up to ~a few hundred MiB for ultracentral collisions)
    # del surface

    #########
    # UrQMD #
    #########

    # hadronic afterburner
    utilities.run_cmd(*['afterburner', 'particles_in.dat', 'particles_out.dat'], quiet=True, deduplicate=True)

    ####################################
    # Post-Hadronic Transport Analysis #
    ####################################

    # read final particle data
    with open(os.path.join(working_dir, 'particles_out.dat'), 'rb') as f:

        # partition UrQMD file into oversamples
        groups = groupby(f, key=lambda l: l.startswith(b'#'))
        samples = filter(lambda g: not g[0], groups)

        # iterate over particles and oversamples
        parts_iter = (
            tuple((nsample, *l.split()))
            for nsample, (header, sample) in enumerate(samples, start=1)
            for l in sample
        )

        try:
            parts = np.fromiter(parts_iter, dtype=parts_dtype)
        except ValueError as e:
            logging.exception(e)
            raise StopEvent('No particles in UrQMD output.')


    # # save raw particle data (optional)
    # # save event to hdf5 data set
    # logging.info('saving raw particle data')
    #
    # particles_file.create_dataset(
    #     'event_{}'.format(event_number),
    #     data=parts, compression='lzf'
    # )

    logging.info('computing observables')
    charged = (parts['charge'] != 0)
    abs_eta = np.fabs(parts['eta'])

    results['dNch_deta'] = \
        np.count_nonzero(charged & (abs_eta < .5)) / nsamples

    ET_eta = .6
    results['dET_deta'] = \
        parts['ET'][abs_eta < ET_eta].sum() / (2 * ET_eta) / nsamples

    abs_ID = np.abs(parts['ID'])
    midrapidity = (np.fabs(parts['y']) < .5)

    pT = parts['pT']
    phi = parts['phi']

    for name, i in species:
        cut = (abs_ID == i) & midrapidity
        N = np.count_nonzero(cut)
        results[f'dN_dy_{name}'] = N / nsamples
        results[f'mean_pT_{name}'] = (0. if N == 0 else pT[cut].mean())

    pT_alice = pT[charged & (abs_eta < .8) & (.15 < pT) & (pT < 2.)]
    results['pT_fluct_N'] = pT_alice.size
    results['pT_fluct_sum_pT'] = pT_alice.sum()
    results['pT_fluct_sum_pTsq'] = np.inner(pT_alice, pT_alice)

    phi_alice = phi[charged & (abs_eta < .8) & (.2 < pT) & (pT < 5.)]
    flow_N = phi_alice.size
    results['flow_N'] = flow_N

    # Add soft flow vectors to result dictionary
    try:
        for n in np.arange(1, 8):  # Add in all flow vectors
            phases = np.exp(1j * n * phi_alice).sum()

            results[f'psi_{n}'] = np.angle(np.real(phases)
                                        + 1j * np.imag(phases))
            results[f'v_{n}'] = np.abs(np.real(phases)
                                    + 1j * np.imag(phases)) / flow_N
    except:
        logging.error('Problem pre-computing v_2 and psi_2!!!')
        pass

    # Save DukeQCD results file
    logging.info('Saving event UrQMD observables...')
    logging.debug(os.getcwd())
    with open("observables.json", "w") as f:
        json.dump(results, f)

    # Open the hydro file and create file object for manipulation.
    logging.info('Creating plasma_event object...')
    plasmaFilePath = os.path.join(working_dir, 'viscous_14_moments_evo.dat')

    # Create event object
    # This asks the hydro file object to interpolate the relevant functions and pass them on to the plasma object.
    event = plasma.plasma_event(hydro_file_path=plasmaFilePath, meta=dict(event_dataframe))

    # Go home & announce
    os.chdir(og_dir)
    logging.info('Event generation complete.')

    return event


# Function to rejection sample a given interpolated temperature function^6 for jet production.
# Returns an accepted (x, y) sample point as a numpy array.
def temp_6th_sample(event, maxAttempts=5, time='i', batch=1000):
    # Get temperature function
    temp_func = event.temp

    # Set time
    np.amin(temp_func.grid[0])
    if time == 'i':
        time = np.amin(temp_func.grid[0])
    elif time == 'f':
        time = np.amax(temp_func.grid[0])
    else:
        pass

    # Find max temp
    maxTemp = event.max_temp(time=time)

    # Find grid bounds
    gridMin = np.amin(temp_func.grid[1])
    gridMax = np.amax(temp_func.grid[1])
    gridWidth = gridMax - gridMin

    attempt = 0
    while attempt < maxAttempts:
        # Generate random point in 3D box of l = w = gridWidth and height maximum temp.^6
        # Origin at center of bottom of box
        pointArray = utilities.cube_random(num = batch, boxSize=gridWidth, maxProb=maxTemp ** 6)

        for point in pointArray:
            targetTemp = temp_func(np.array([time, point[0], point[1]]))**6

            # Check if point under 2D temp PDF curve
            if float(point[2]) < float(targetTemp[0]):
                # If under curve, accept point and return
                # print("Attempt " + str(attempt) + " successful with point " + str(i) + "!!!")
                # print(point)
                # print("Random height: " + str(zPoints[i]))
                # print("Target <= height: " + str(float(targetTemp)))
                return point[0:2]
        logging.info("Jet Production Sampling Attempt: " + str(attempt) + " failed.")
        attempt += 1
    logging.error("Error in jet production point sampling!")
    return np.array([0,0,0])


# Function to generate a given number of jet production points
# sampled from the temperature^6 profile.
def generate_jet_seed_point(event, num=1):
    pointArray = np.array([])
    for i in np.arange(0, num):
        newPoint = temp_6th_sample(event)
        if i == 0:
            pointArray = newPoint
        else:
            pointArray = np.vstack((pointArray, newPoint))
    return pointArray


# Function to create Woods-Saxon distribution initial conditions
def woods_saxon_ic(b, A=208, R=6.62, a=0.546, p=-1, norm=1,
                   grid_step=config.transport.GRID_STEP, rmax=config.transport.GRID_MAX_TARGET):
    # Defaults are Trento PbPb parameters

    # Determine radius
    if R == None:
        R = 1.25 * (A)**(1/3)  # Good approximation, re:https://en.wikipedia.org/wiki/Woods%E2%80%93Saxon_potential

    # Define ic function
    ws = lambda x, y, z, x0: 1 / (1 + np.exp( (np.sqrt((x-x0)**2 + y**2 + z**2) - R) / a))
    z_vals = np.arange(-R, R, 0.5)
    # See Trento discussion, eq. 5: https://arxiv.org/abs/1412.4708
    if p == 0:
        TR_func = np.vectorize(lambda x, y: (integrate.trapezoid(ws(x, y, z_vals, -b / 2))
                                        * integrate.trapezoid(ws(x, y, z_vals, b / 2))) ** (1 / 2))
    elif p == 'bc':  # Binary colllision scaling
        TR_func = np.vectorize(lambda x, y: (integrate.trapezoid(ws(x, y, z_vals, -b / 2))
                                        * integrate.trapezoid(ws(x, y, z_vals, b / 2))))
    else:
        TR_func = np.vectorize(lambda x, y: (integrate.trapezoid(ws(x, y, z_vals, -b / 2))**p
                                                         + integrate.trapezoid(ws(x, y, z_vals, b / 2))**p)**(1/p))

    # Compute tabulated values for function


    # Create meshgrid and evaluate TR function for given points
    x_space = np.arange((0 - rmax), rmax, grid_step)
    x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
    TR = TR_func(x_coords, y_coords)

    # Normalize and transpose (to set impact parameter along x-axis)
    array = np.transpose((norm/(np.sum(TR) * (grid_step ** 2))) * TR)

    return array, grid_step

# Function to create plasma object for Woods-Saxon distribution
# Alpha is expansion power level --
def woods_saxon_plasma(b, T0=0.39, A=208, a=0.546, alpha=0, name=None,
                       resolution=5, rmax=10, tmin=0.5, tmax=None, umax=0.75, return_grids=False):
    # Defaults are Trento PbPb parameters

    # Determine radius
    R = 1.25 * (A)**(1/3)  # Good approximation, re:https://en.wikipedia.org/wiki/Woods%E2%80%93Saxon_potential

    # Define temperature and velocity functions
    ws = lambda x, y, z, x0: 1 / (1 + np.exp( (np.sqrt((x-x0)**2 + y**2 + z**2) - R) / a))

    # Define grid time and space domains
    if tmax is None:
        tmax = 2 * rmax
    t_space = np.linspace(tmin, tmax, int((rmax + rmax) * resolution))
    x_space = np.linspace((0 - rmax), rmax, int((rmax + rmax) * resolution))
    grid_step = (2 * rmax) / int((rmax + rmax) * resolution)

    # Create meshgrid for function evaluation
    t_coords, x_coords, y_coords = np.meshgrid(t_space, x_space, x_space, indexing='ij')
    z_vals = np.arange(-R, R, 0.5)

    # Compute overlap function
    TATB = np.vectorize(lambda t, x, y: integrate.trapezoid(ws(x, y, z_vals, -b / 2))
                                                     * integrate.trapezoid(ws(x, y, z_vals, b / 2))
                                                     * ((tmin / t) ** (alpha)))


    # Evaluate functions for grid points
    temp_values = (T0 / 2.708) * np.power(TATB(t_coords, x_coords, y_coords), 1/6)
    temp_values =  temp_values  # normalize max temp to proper event

    temp_grad_x_values = np.gradient(temp_values, grid_step, axis=1)
    temp_grad_y_values = np.gradient(temp_values, grid_step, axis=2)

    max_grad_mag = np.amax(np.sqrt(temp_grad_x_values ** 2 + temp_grad_y_values ** 2))

    # Velocities are proportional to negative temperature gradient.
    x_vel_values = (-1) * umax * np.gradient(temp_values, grid_step, axis=1) / max_grad_mag
    y_vel_values = (-1) * umax * np.gradient(temp_values, grid_step, axis=2) / max_grad_mag

    logging.info(np.ndim(temp_values))

    # Create and return plasma object
    plasma_object = plasma.tabulated_plasma(t_space, x_space, temp_values, x_vel_values, y_vel_values, name=name,
                            return_grids=return_grids)

    # Return the grids of evaluated points, if requested.
    if return_grids:
        return plasma_object, temp_values, x_vel_values, y_vel_values,
    else:
        return plasma_object


# Function to create plasma object for Woods-Saxon distribution
# Alpha is expansion power level --
def gaussian_plasma(b, T0=0.39, A=208, alpha=0.5, name=None,
                       resolution=5, rmax=10, tmin=0.5, tmax=None, umax=0.75, return_grids=False):
    # Defaults are Trento PbPb parameters

    # Determine radius
    R = float(1.25 * (A)**(1/3))  # Good approximation, re:https://en.wikipedia.org/wiki/Woods%E2%80%93Saxon_potential

    # Define grid time and space domains
    if tmax is None:
        tmax = 2 * rmax
    t_space = np.linspace(tmin, tmax, int((rmax + rmax) * resolution))
    x_space = np.linspace((0 - rmax), rmax, int((rmax + rmax) * resolution))
    grid_step = (2 * rmax) / int((rmax + rmax) * resolution)

    # Create meshgrid for function evaluation
    t_coords, x_coords, y_coords = np.meshgrid(t_space, x_space, x_space, indexing='ij')

    # Compute overlap function
    TATB = np.vectorize(lambda t, x, y: np.exp(- x**2 / ((2*R - b)**2)) * np.exp(- y**2 / ((2*R)**2))
                                                     * ((tmin / t) ** (alpha)))


    # Evaluate functions for grid points
    # temp_values = np.power(np.multiply(T_A(t_coords, x_coords, y_coords),
    #                                    T_B(t_coords, x_coords, y_coords)), 1/2)
    temp_values = TATB(t_coords, x_coords, y_coords)
    temp_values = temp_values * (T0 / np.amax(temp_values)) # normalize max temp to proper event

    temp_grad_x_values = np.gradient(temp_values, grid_step, axis=1)
    temp_grad_y_values = np.gradient(temp_values, grid_step, axis=2)

    max_grad_mag = np.amax(np.sqrt(temp_grad_x_values ** 2 + temp_grad_y_values ** 2))

    # Velocities are proportional to negative temperature gradient.
    x_vel_values = (-1) * umax * np.gradient(temp_values, grid_step, axis=1) / max_grad_mag
    y_vel_values = (-1) * umax * np.gradient(temp_values, grid_step, axis=2) / max_grad_mag

    logging.info(np.ndim(temp_values))
    # Compute gradients


    # Create and return plasma object
    plasma_object = plasma.tabulated_plasma(t_space, x_space, temp_values, x_vel_values, y_vel_values, name=name,
                            return_grids=return_grids)

    # Return the grids of evaluated points, if requested.
    if return_grids:
        return plasma_object, temp_values, x_vel_values, y_vel_values,
    else:
        return plasma_object