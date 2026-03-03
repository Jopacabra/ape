import logging
import math
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import config

# Command to run process in the terminal
# Stolen and modified from DukeQCD "run-events.py":
# https://github.com/Duke-QCD/hic-eventgen
def run_cmd(*args, quiet=False):
    """
    Run and log a Subprocess.
    """
    cmd = ' '.join(args)
    logging.info('running command: {}'.format(cmd))
    processName = str(args[0])

    try:
        proc = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            'command failed with status {}}:\n{}}'.format(e.returncode, e.output.strip('\n'))
        )
        raise
    else:
        logging.debug(
            'command completed successfully:\n{}'.format(proc.stdout)
        )
        outputArray = np.array([])
        if not quiet:
            outputCopyStdout = proc.stdout
            outputCopyStderr = proc.stderr
            # Attempt to print the output from the trentoSubprocess.
            logging.info('------------- {} Output ----------------'.format(processName))
            logging.debug('exit status:\n{}'.format(proc.returncode))
            logging.info('stdout:\n')
            for line in outputCopyStdout:
                logging.info(line)
                outputArray = np.append(outputArray, line)
            logging.info('stderr:\n')
            while True:
                try:
                    line = outputCopyStderr.readline()
                except AttributeError:
                    break
                logging.debug(line)
                if not line:
                    break
            logging.info('----------------------------------------')

        return proc, outputArray


# Function to round up to specified number of decimals
def round_decimals_up(number: float, decimals: int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


# Function to round down to specified number of decimals
def round_decimals_down(number: float, decimals: int = 1):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


# Creates a temporary directory and moves to it.
# Returns tempfile.TemporaryDirectory object.
def tempDir(location=None):
    # Get current directory if no location supplied
    if location is None:
        location = os.getcwd()
    # Create and move to temp directory
    temp_dir = tempfile.TemporaryDirectory(prefix='JMA_', dir=str(location))
    logging.info('Created temp directory {}'.format(temp_dir.name))
    os.chdir(temp_dir.name)

    return temp_dir


# Generate a random (x, y, z) coordinate in a 3D box of l = w = boxSize and h = maxProb
# Origin at cent of bottom of box.
def cube_random(num=1, boxSize=1, maxProb=1):
    rng = np.random.default_rng()
    pointArray = np.array([])
    for i in np.arange(0, num):
        x = (boxSize * rng.random()) - (boxSize / 2)
        y = (boxSize * rng.random()) - (boxSize / 2)
        z = maxProb * rng.random()
        newPoint = np.array([x,y,z])
        if i == 0:
            pointArray = newPoint
        else:
            pointArray = np.vstack((pointArray, newPoint))
    return pointArray


# Generate a random (x, y) coordinate in a 2D box of w = boxSize and h = maxProb
# Origin at bottom left of box.
def random_2d(num=1, boxSize=1.0, maxProb=1.0):
    rng = np.random.default_rng()
    pointArray = np.array([])
    for i in np.arange(0, num):
        x = boxSize * rng.random()
        y = maxProb * rng.random()
        newPoint = np.array([x,y])
        if i == 0:
            pointArray = newPoint
        else:
            pointArray = np.vstack((pointArray, newPoint))
    return pointArray


# Function to rejection sample E^{-4} dist. for jet energy selection.
def jet_e_sample(maxAttempts=5, batch=1000, min_e=0, max_e=100):

    attempt = 0
    while attempt < maxAttempts:
        # Generate random point
        pointArray = random_2d(num=batch, boxSize=max_e, maxProb=1)

        for point in pointArray:
            if point[0] > min_e:
                targetE = point[0] ** (-4)

                # Check if point under E PDF curve
                if float(point[1]) < float(targetE):
                    # If under curve, accept point and return
                    # print("Attempt " + str(attempt) + " successful with point " + str(i) + "!!!")
                    # print(point)
                    # print("Random height: " + str(zPoints[i]))
                    # print("Target <= height: " + str(float(targetTemp)))
                    return point[0]
        #print("Jet Energy Sampling Attempt: " + str(attempt) + " failed.")
        attempt += 1
    print("Catastrophic error in jet energy sampling!")
    print("AHHHHHHHHHHHHHHH!!!!!!!!!!!")
    return 0

# Function generally used to average a medium parameter over a certain pathlength
def dtau_avg(func, point, phi, dtau, beta, num_samples=10):
    sample_coords = point
    for delta_tau in np.arange(dtau/num_samples, dtau, dtau/num_samples):
        sample_tau = point[0] + delta_tau
        sample_x = point[1] + (beta * delta_tau * np.cos(phi))
        sample_y = point[2] + (beta * delta_tau * np.sin(phi))
        sample_coords = np.vstack((sample_coords, np.array([sample_tau, sample_x, sample_y])))

    # Return zero if any point within the step would be out of bounds
    try:
        value = np.mean(func(sample_coords))
    except ValueError:
        value = 0

    # return averaged value
    return value

# Function to take an IC object and return the angle for epsilon_n
def ecc_more(ic, n):
    r"""
    Calculate the angle and magnitude of the eccentricity harmonic `\varepsilon_n`.

    :param int n: Eccentricity order.

    """
    ny, nx = ic._profile.shape
    xmax, ymax = ic._xymax
    xcm, ycm = ic._cm

    # create (X, Y) grids relative to CM
    Y, X = np.mgrid[ymax:-ymax:1j*ny, -xmax:xmax:1j*nx]
    X -= xcm
    Y -= ycm

    # create grid of weights = profile * R^n
    Rsq = X*X + Y*Y
    if n == 1:
        W = np.sqrt(Rsq, out=Rsq)
    elif n == 2:
        W = Rsq
    else:
        if n & 1:  # odd n
            W = np.sqrt(Rsq)
        else:  # even n
            W = np.copy(Rsq)
        # multiply by R^2 until W = R^n
        for _ in range(int((n-1)/2)):
            W *= Rsq
    W *= ic._profile

    # create grid of e^{i*n*phi} * W
    i_n_phi = np.zeros_like(X, dtype=complex)
    np.arctan2(Y, X, out=i_n_phi.imag)
    i_n_phi.imag *= n
    exp_phi = np.exp(i_n_phi, out=i_n_phi)
    exp_phi *= W

    return abs(exp_phi.sum()) / W.sum(), np.angle(exp_phi.sum() / W.sum())

# Function that takes a pandas dataframe and creates a histogramed weight xarray
def xarray_ify(df, pt_series='pt_f', phi_series='phi_f', pid_series=None, weight_series='AA_weight',
               drift=True, cel=False, NUM_PHI=157, K_F_DRIFT=1.0):
    if not drift:
        K_F_DRIFT = 0.0

    # Make cut
    mask = (df['drift'] == drift) & (df['K_F_DRIFT'] == K_F_DRIFT) & (df['cel'] == cel)

    # Get list of ids
    id_list = df['id'].value_counts().index

    # Select bins for the coordinates
    pt_bins = np.arange(0, 101, config.jet.PT_BIN)
    pt_bin_labels = (pt_bins[1:] + pt_bins[0:-1]) / 2

    phi_bins = np.linspace(0, 2 * np.pi, NUM_PHI)
    phi_bin_labels = (phi_bins[1:] + phi_bins[0:-1]) / 2

    pid_bins = np.array([-3.5, -2.5, -1.5, -0.5, 1.5, 2.5, 3.5, 21.5])
    pid_bin_labels = np.array([-3, -2, -1, 1, 2, 3, 21])

    # Get the lists of coordinates
    pt_array = df[mask][pt_series].to_numpy()
    phi_array = df[mask][phi_series].to_numpy()
    weights = df[mask][weight_series].to_numpy()

    # Histogram with PIDs, if requested
    if pid_series is not None:
        # Get pid coordinates
        pid_array = df[mask][pid_series].to_numpy()


        # Zip them up in ordered pair coordinates
        coords = np.stack([pt_array, phi_array, pid_array], axis=1)

        # Do 4D histogram
        H, edges = np.histogramdd(coords, bins=(pt_bins, phi_bins, pid_bins), weights=weights, density=False)

        # Make it an xarray DataArray
        xr_hist = xr.DataArray(H, coords={"pt": pt_bin_labels, "phi": phi_bin_labels, "pid": pid_bin_labels})

    else:
        # Zip them up in ordered pair coordinates
        coords = np.stack([pt_array, phi_array], axis=1)

        # Do 4D histogram
        H, edges = np.histogramdd(coords, bins=(pt_bins, phi_bins), weights=weights, density=False)

        # Make it an xarray DataArray
        xr_hist = xr.DataArray(H, coords={"pt": pt_bin_labels, "phi": phi_bin_labels})

    # Record xarray attributes
    event_psi_2 = df.loc[mask, 'psi_2'].to_numpy()[0]
    event_mult = df.loc[mask, 'mult'].to_numpy()[0]
    event_e_2 = df.loc[mask, 'e2'].to_numpy()[0]
    event_v_2 = df.loc[mask, 'v_2'].to_numpy()[0]
    event_Tmax = df.loc[mask, 'Tmax_event'].to_numpy()[0]
    xr_hist.attrs['psi_2'] = event_psi_2
    xr_hist.attrs['e_2'] = event_e_2
    xr_hist.attrs['v_2'] = event_v_2
    xr_hist.attrs['mult'] = event_mult
    xr_hist.attrs['Tmax'] = event_Tmax
    xr_hist.attrs['drift'] = drift
    xr_hist.attrs['cel'] = cel
    xr_hist.attrs['K_F_Drift'] = K_F_DRIFT

    return xr_hist

# Function to package one event's output into histogrammed xarray files of fragmented hadrons
def xarray_ify_ff(df, pt_series='pt_f', phi_series='phi_f', z_series='z', weight_series='AA_weight',
               drift=True, cel=False, NUM_PHI=157, K_F_DRIFT=1.0):
    if not drift:
        K_F_DRIFT = 0.0
    # Select bins for the coordinates
    pt_bins = np.arange(0, 101, config.jet.PT_BIN)
    pt_bin_labels = (pt_bins[1:] + pt_bins[0:-1]) / 2

    phi_bins = np.linspace(0, 2 * np.pi, NUM_PHI)
    phi_bin_labels = (phi_bins[1:] + phi_bins[0:-1]) / 2

    # Cut to case & get info
    loaded_pd_partons = df[(df['K_F_DRIFT'] == K_F_DRIFT)
                           & (df['drift'] == drift) & (df['cel'] == cel)]
    event_psi_2 = loaded_pd_partons.loc[:, 'psi_2'].to_numpy()[0]
    event_mult = loaded_pd_partons.loc[:, 'mult'].to_numpy()[0]
    event_e_2 = loaded_pd_partons.loc[:, 'e2'].to_numpy()[0]
    event_v_2 = loaded_pd_partons.loc[:, 'v_2'].to_numpy()[0]
    event_Tmax = loaded_pd_partons.loc[:, 'Tmax_event'].to_numpy()[0]

    # Get the lists of coordinates
    # Find out how many fragmentations were done:
    num_frags = len(loaded_pd_partons['z'].to_numpy()[0])

    # Get ordered list of fragmentations
    z_vals = np.concatenate(loaded_pd_partons[z_series].to_numpy(), axis=0)
    part_pt_array = np.repeat(loaded_pd_partons[pt_series].to_numpy(), repeats=num_frags)
    had_pt_array = z_vals * part_pt_array
    phi_array = np.repeat(loaded_pd_partons[phi_series].to_numpy(), repeats=num_frags)
    weights = np.repeat(loaded_pd_partons[weight_series].to_numpy(), repeats=num_frags)

    # Zip them up in ordered pair coordinates
    coords = np.stack([had_pt_array, phi_array], axis=1)

    # Do 3D histogram
    H, edges = np.histogramdd(coords, bins=(pt_bins, phi_bins), weights=weights, density=False)

    # Make it an xarray DataArray
    xr_hist = xr.DataArray(H, coords={"pt": pt_bin_labels, "phi": phi_bin_labels})

    # Record xarray attributes
    xr_hist.attrs['psi_2'] = event_psi_2
    xr_hist.attrs['e_2'] = event_e_2
    xr_hist.attrs['v_2'] = event_v_2
    xr_hist.attrs['mult'] = event_mult
    xr_hist.attrs['Tmax'] = event_Tmax
    xr_hist.attrs['drift'] = drift
    xr_hist.attrs['cel'] = cel
    xr_hist.attrs['K_F_Drift'] = K_F_DRIFT

    # Save some memory
    del loaded_pd_partons
    del part_pt_array
    del had_pt_array
    del phi_array
    del weights

    return xr_hist


# Function to package many events' output into histogrammed xarray files of partons
def xarray_ify_many(df, pt_series='pt_f', phi_series='phi_f', pid_series=None, weight_series='AA_weight',
               drift=True, cel=False, NUM_PHI=157, K_F_DRIFT=1.0):
    if not drift:
        K_F_DRIFT = 0.0
    # Select bins for the coordinates
    pt_bins = np.arange(0, 101, config.jet.PT_BIN)
    pt_bin_labels = (pt_bins[1:] + pt_bins[0:-1]) / 2

    phi_bins = np.linspace(0, 2 * np.pi, NUM_PHI)
    phi_bin_labels = (phi_bins[1:] + phi_bins[0:-1]) / 2

    id_bins = np.array([-3.5, -2.5, -1.5, -0.5, 1.5, 2.5, 3.5, 21.5])
    id_bin_labels = np.array([-3, -2, -1, 1, 2, 3, 21])

    # Iterate through events and combine to one dataset
    xr_all = xr.Dataset({})
    seed_list = df['seed'].value_counts().index.to_numpy()
    for i in np.arange(0, len(seed_list)):
        #print(i / len(seed_list))
        # Cut to one event
        seed = seed_list[i]
        loaded_pd_partons = df[(df['seed'] == seed) & (df['K_F_DRIFT'] == K_F_DRIFT)
                               & (df['drift'] == drift) & (df['cel'] == cel)]
        event_psi_2 = loaded_pd_partons.loc[:, 'psi_2'].to_numpy()[0]
        event_mult = loaded_pd_partons.loc[:, 'mult'].to_numpy()[0]
        event_e_2 = loaded_pd_partons.loc[:, 'e2'].to_numpy()[0]
        event_v_2 = loaded_pd_partons.loc[:, 'v_2'].to_numpy()[0]
        event_Tmax = loaded_pd_partons.loc[:, 'Tmax_event'].to_numpy()[0]

        # Get the lists of coordinates
        pt_array = loaded_pd_partons[pt_series].to_numpy()
        phi_array = loaded_pd_partons[phi_series].to_numpy()
        pid_array = loaded_pd_partons[pid_series].to_numpy()
        weights = loaded_pd_partons[weight_series].to_numpy()

        # Zip them up in ordered pair coordinates
        coords = np.stack([pt_array, phi_array, pid_array], axis=1)

        # Do 3D histogram
        H, edges = np.histogramdd(coords, bins=(pt_bins, phi_bins, id_bins), weights=weights, density=False)

        # Make it an xarray DataArray
        xr_hist = xr.DataArray(H, coords={"pt": pt_bin_labels, "phi": phi_bin_labels, "pid": id_bin_labels})

        # Record psi_2 as xarray attribute
        xr_hist.attrs['psi_2'] = event_psi_2
        xr_hist.attrs['e_2'] = event_e_2
        xr_hist.attrs['v_2'] = event_v_2
        xr_hist.attrs['mult'] = event_mult
        xr_hist.attrs['Tmax'] = event_Tmax
        xr_hist.attrs['drift'] = drift
        xr_hist.attrs['cel'] = cel
        xr_hist.attrs['K_F_Drift'] = K_F_DRIFT

        # Add to growing list of events
        xr_all[str(seed)] = xr_hist

        # Save some memory
        del loaded_pd_partons
        del pt_array
        del phi_array
        del pid_array
        del weights

    return xr_all


# Function to package many events' output into histogrammed xarray files of fragmented hadrons
def xarray_ify_many_ff(df, pt_series='pt_f', phi_series='phi_f', z_series='z', weight_series='AA_weight',
               drift=True, cel=False, NUM_PHI=157, K_F_DRIFT=1.0):
    if not drift:
        K_F_DRIFT = 0.0
    # Select bins for the coordinates
    pt_bins = np.arange(0, 101, config.jet.PT_BIN)
    pt_bin_labels = (pt_bins[1:] + pt_bins[0:-1]) / 2

    phi_bins = np.linspace(0, 2 * np.pi, NUM_PHI)
    phi_bin_labels = (phi_bins[1:] + phi_bins[0:-1]) / 2

    # Iterate through events and combine to one dataset
    xr_all = xr.Dataset({})
    seed_list = df['seed'].value_counts().index.to_numpy()
    for i in np.arange(0, len(seed_list)):
        #print(i / len(seed_list))
        # Cut to one event
        seed = seed_list[i]
        loaded_pd_partons = df[(df['seed'] == seed) & (df['K_F_DRIFT'] == K_F_DRIFT)
                               & (df['drift'] == drift) & (df['cel'] == cel)]
        event_psi_2 = loaded_pd_partons.loc[:, 'psi_2'].to_numpy()[0]
        event_mult = loaded_pd_partons.loc[:, 'mult'].to_numpy()[0]
        event_e_2 = loaded_pd_partons.loc[:, 'e2'].to_numpy()[0]
        event_v_2 = loaded_pd_partons.loc[:, 'v_2'].to_numpy()[0]
        event_Tmax = loaded_pd_partons.loc[:, 'Tmax_event'].to_numpy()[0]

        # Get the lists of coordinates
        # Find out how many fragmentations were done:
        num_frags = len(loaded_pd_partons['z'].to_numpy()[0])

        # Get ordered list of fragmentations
        z_vals = np.concatenate(loaded_pd_partons[z_series].to_numpy(), axis=0)
        part_pt_array = np.repeat(loaded_pd_partons[pt_series].to_numpy(), repeats=num_frags)
        had_pt_array = z_vals * part_pt_array
        phi_array = np.repeat(loaded_pd_partons[phi_series].to_numpy(), repeats=num_frags)
        weights = np.repeat(loaded_pd_partons[weight_series].to_numpy(), repeats=num_frags)

        # Zip them up in ordered pair coordinates
        coords = np.stack([had_pt_array, phi_array], axis=1)

        # Do 3D histogram
        H, edges = np.histogramdd(coords, bins=(pt_bins, phi_bins), weights=weights, density=False)

        # Make it an xarray DataArray
        xr_hist = xr.DataArray(H, coords={"pt": pt_bin_labels, "phi": phi_bin_labels})

        # Record psi_2 as xarray attribute
        xr_hist.attrs['psi_2'] = event_psi_2
        xr_hist.attrs['e_2'] = event_e_2
        xr_hist.attrs['v_2'] = event_v_2
        xr_hist.attrs['mult'] = event_mult
        xr_hist.attrs['Tmax'] = event_Tmax
        xr_hist.attrs['drift'] = drift
        xr_hist.attrs['cel'] = cel
        xr_hist.attrs['K_F_Drift'] = K_F_DRIFT

        # Add to growing list of events
        xr_all[str(seed)] = xr_hist

        # Save some memory
        del loaded_pd_partons
        del part_pt_array
        del had_pt_array
        del phi_array
        del weights

    return xr_all


# Get the component of vector a perpendicular to vector b.
def perp_vec(a, b):
    # Get numpy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Magnitude of p
    pp = np.dot(b, b)

    # Subtract off component of u perpendicular to p
    return a - (np.dot(a, b) / pp) * b

# Get the component of vector a parallel to vector b.
def par_vec(a, b):
    # Get numpy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Magnitude of p
    pp = np.dot(b, b)

    # Component of u parallel to p
    return (np.dot(a, b) / pp) * b


def zeta(q=0, maxAttempts=5, batch=1000):
    # Special cases making things easier
    if q == 0:
        rng = np.random.default_rng()
        return rng.random() * 2
    elif q == -1:
        return 1

    attempt = 0
    while attempt < maxAttempts:
        # Generate random point in 3D box of l = w = gridWidth and height maximum temp.^6
        # Origin at center of bottom of box
        pointArray = utilities.random_2d(num=batch, boxSize=q + 2, maxProb=1)
        for point in pointArray:
            x = point[0]
            y = point[1]
            targetVal = ((1 + q) / ((q + 2) ** (1 + q))) * ((q + 2 - x) ** q)

            # Check if point under 2D temp PDF curve
            if float(y) < float(targetVal):
                # If under curve, accept point and return
                # print("Attempt " + str(attempt) + " successful with point " + str(i) + "!!!")
                # print(point)
                # print("Random height: " + str(zPoints[i]))
                # print("Target <= height: " + str(float(targetTemp)))
                return x
        print("Zeta Parameter Sample Attempt: " + str(attempt) + " failed.")
        attempt += 1
    print("Catastrophic error in zeta parameter sampling!")
    print("AHHHHHHHHHHHHHHH!!!!!!!!!!!")
    return 0
