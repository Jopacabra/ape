import logging
import math
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import config

# Create global rng
rng = np.random.default_rng(seed=config.mode.SEED)

# Command to run process in the terminal
# Stolen and modified from DukeQCD "run-events.py":
# https://github.com/Duke-QCD/hic-eventgen
def run_cmd(*args, quiet=False, deduplicate=False):
    """
    Run and log a Subprocess.

    Uses communicate() for safe, deadlock-free process handling.
    All output is captured and available after process completion.
    """
    cmd = ' '.join(args)
    logging.info('running shell command:\n{}'.format(cmd))
    processName = str(args[0])

    try:
        proc = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            'command failed with status {}}:\n{}}'.format(e.returncode, e.output.strip('\n'))
        )
        raise

    # communicate() waits for process to complete and captures all output safely
    # This avoids deadlocks that can occur with manual pipe reading
    output_text, _ = proc.communicate()

    # Parse output into lines as a numpy array
    outputArray = np.array([line for line in output_text.split('\n') if line])

    if quiet:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if deduplicate:  # Special case for urqmd: trim duplicate lines and log to debug

        unique_lines = []
        seen = set()

        for line in outputArray:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)
        logging.log(level, '------------- {} Output Start (Deduplicated) ----------------'.format(processName))
        logging.log(level, 'exit status: {}'.format(proc.returncode))
        logging.log(level, 'stdout (showing {} unique lines of {} total):'.format(len(unique_lines), len(outputArray)))
        for line in unique_lines:
            logging.log(level, line)
        logging.log(level, '------------- {} Output End ----------------'.format(processName))
    else:
        logging.log(level, '------------- {} Output Start ----------------'.format(processName))
        logging.log(level, 'exit status: {}'.format(proc.returncode))
        logging.log(level, 'stdout:')
        for line in outputArray:
            logging.log(level, line)
        logging.log(level, '------------- {} Output End ----------------'.format(processName))

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
def cube_random(num=1, boxSize=1, maxProb=1, seed=None):
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


# Get the component of vector a perpendicular to vector b.
def perp_vec(a, b):
    # Get numpy arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Magnitude^2 of p
    pp = np.dot(b, b)

    # Subtract off component of a perpendicular to b
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


def monte_carlo_causal_sphere_integral(interpolator, ref, dt, n_samples=10000):
    """
    Perform Monte Carlo integration over the causal sphere using a RegularGridInterpolator.

    The causal sphere is defined by the lightlike condition:
    0 = (t - t_ref)^2 - (x - x_ref)^2 - (y - y_ref)^2 - (z - z_ref)^2

    Within a small time interval dt, we approximate the integral by sampling uniformly
    on the future lightcone surface at time t_ref + dt.

    Parameters
    ----------
    interpolator : scipy.interpolate.RegularGridInterpolator
        The interpolator object for the medium properties
    t_ref : float
        Reference time coordinate
    x_ref : float
        Reference x coordinate
    y_ref : float
        Reference y coordinate
    z_ref : float
        Reference z coordinate
    dt : float
        Small time interval over which to integrate
    n_samples : int, optional
        Number of Monte Carlo samples to draw (default: 10000)

    Returns
    -------
    float
        The Monte Carlo estimate of the integral
    """
    t_ref, x_ref, y_ref, z_ref = ref

    # Time coordinate on the causal sphere
    t_new = t_ref + dt

    # On the lightcone: dt^2 = dx^2 + dy^2 + dz^2
    # So points on the sphere have distance t_ref from the reference point
    # We integrate over points that could have sent a signal that arrives at time t_ref at this spatial point.
    sphere_radius = t_ref

    # Generate random points uniformly on a sphere
    # Use the standard method: random direction on unit sphere scaled by radius
    samples = np.random.normal(size=(n_samples, 3))

    # Normalize to unit sphere, then scale by the causal radius
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    directions = samples / norms
    sphere_points = directions * sphere_radius

    # Translate to actual coordinates centered at (x_ref, y_ref, z_ref)
    x_samples = x_ref + sphere_points[:, 0]
    y_samples = y_ref + sphere_points[:, 1]
    z_samples = z_ref + sphere_points[:, 2]

    # Stack coordinates for interpolator evaluation
    # RegularGridInterpolator expects points as (n_points, n_dims)
    evaluation_points = np.stack([
        np.full(n_samples, t_new),
        x_samples,
        y_samples,
        z_samples
    ], axis=1)

    # Evaluate the interpolator at all points
    values = interpolator(evaluation_points)

    # The surface element on a sphere in 3D is: dS = R^2 sin(theta) dtheta dphi
    # For a sphere of radius R, the total surface area is 4*pi*R^2
    # The integral is: integral = (1/N) * sum(f(p_i)) * surface_area
    surface_area = 4 * np.pi * sphere_radius ** 2

    # Monte Carlo estimate
    integral_estimate = np.mean(values) * surface_area

    return integral_estimate


def config_to_dict(cls, prefix=""):
    """
    Flatten the contents of the nested config class into a dictionary.
    """
    result = {}
    for key in dir(cls):
        if key.startswith("_"):
            continue
        value = getattr(cls, key)
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, type):  # It's a nested class
            result.update(config_to_dict(value, prefix=full_key))
        else:
            result[full_key] = value
    return result