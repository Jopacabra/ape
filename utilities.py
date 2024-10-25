import logging
import math
import os
import subprocess
import tempfile

import numpy as np


# Command to run process in the terminal
# Stolen and modified from DukeQCD "run-events.py":
# https://github.com/Duke-QCD/hic-eventgen
import pandas as pd


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


# Function to create empty results dataframe.
def resultsFrameOG():
    resultsDataframe = pd.DataFrame(
            {
                "eventNo": [],
                "jetNo": [],
                "jet_pT": [],
                "q_BBMG": [],
                "q_BBMG_err": [],
                "q_drift_plasma": [],
                "q_drift_plasma_err": [],
                "q_drift_hrg": [],
                "q_drift_hrg_err": [],
                "q_drift_unhydro": [],
                "q_drift_unhydro_err": [],
                "k_moment": [],
                "shower_correction": [],
                "X0": [],
                "Y0": [],
                "theta0": [],
                "tau_unhydro": [],
                "tau_hrg": [],
                "t_total_plasma": [],
                "t_total_hrg": [],
                "t_total_unhydro": [],
                "Tmax_jet": [],
                "initial_time": [],
                "final_time": [],
                "dx": [],
                "dt": [],
                "rmax": [],
                "Tmax_event": [],
                "b": [],
                "R": [],
                "e2": [],
                "mult": [],
                "phi_2": [],
            }
        )

    return resultsDataframe


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


