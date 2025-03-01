import numpy as np
import pandas as pd
try:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tkr
except:
    print('NO MATPLOTLIB')
from scipy.interpolate import RegularGridInterpolator
import config
import logging


class osu_hydro_file:
    def __init__(self, file_path, event_name=None, temp_conv_factor=0.1973269788):
        # Store your original file location and event number
        self.file_path = file_path
        self.name = event_name

        # Announce initialization
        print('Reading osu-hydro file ... event: ' + str(self.name))

        # Store grid data
        self.grid_data = pd.read_table(file_path, header=None, sep='\s+', dtype=np.float64,
                                       names=['time', 'xpos', 'ypos', 'temp', 'xvel', 'yvel'])

        # Initialize all the ordinary grid parameters

        # Grid is always square. Number of lines of the same time is the number of grid squares == grid_width**2.
        # Note that we check the END timestep because the first timestep may be repeated
        self.grid_width = int(np.sqrt(self.grid_data[['time']].value_counts().to_numpy()[-1]))

        # n_grid_spaces is total number of grid spaces / bins. Assumes square grid.
        self.n_grid_spaces = self.grid_width ** 2

        # Number of time steps is grid_data's time column divided by number of grid spaces.
        self.NT = int(len(self.grid_data[['time']]) / self.n_grid_spaces)

        # Set grid space & time domains & lists

        # We get lists of time and space coordinates from the file
        # These are literally just numpy arrays of the xpos and time columns
        # Use this to match up to NT x grid_width x grid_width array of values for interpolation
        self.xlist = np.ndarray.flatten(pd.DataFrame(self.grid_data[['xpos']]).to_numpy())
        self.tlist = np.ndarray.flatten(pd.DataFrame(self.grid_data[['time']]).to_numpy())

        # Difference in absolute time between steps in simulation
        # Note that we find the timestep from the end of the list. In files from osu-hydro,
        # the first two timesteps are labeled with the same absolute time.
        self.timestep = self.tlist[-1] - self.tlist[-1 - int(self.n_grid_spaces)]

        # Difference in absolute space between grid positions
        # Note that we want a real, positive value for this
        self.gridstep = np.abs(self.xlist[-1] - self.xlist[-2])

        # Domains of physical times and positions
        """
        Absolute time for the first two steps from osu-hydro are the same.
        If we shift the first time step's absolute time back by the difference in timesteps,
        we can relabel the first step and keep the absolute times matching for every other step.

        If we do nothing, creating this linear time space will smoosh an extra timestep into the
        same initial and final time bounds. This, in principle, causes the evolution to happen 
        in the interpolated functions ever so slightly faster than it actually does in osu-hydro.
        This may be a small difference, but it is certainly non-physical.

        For now, we choose to shift back the first timestep by default.
        """
        # Domain of corrected time values
        self.tspace = np.linspace(np.amin(self.tlist) - self.timestep, np.amax(self.tlist), self.NT)
        # Domain of space values
        self.xspace = np.linspace(np.amin(self.xlist), np.amax(self.xlist), self.grid_width)

        # Determine minimum and maximum grid values
        self.gridMin = np.amin(self.xspace)
        self.gridMax = np.amax(self.xspace)

        # Determine initial and final times
        self.t0 = np.amin(self.tspace)
        self.tf = np.amax(self.tspace)

        # Record conversion factor for temperature
        # Converts temperature to GeV -- By default goes from fm^-1 to GeV
        self.temp_conv_factor = temp_conv_factor

    # Method to get raw temp data
    def temp_array(self):
        # Cut temp data out and convert to numpy array
        temp_column = self.grid_data[['temp']]
        temp_data = pd.DataFrame(temp_column).to_numpy()

        # Reshape this nonsense into an array with first index time, then x, then y.
        # You can get the temperature at the grid indexes (ix,iy) at timestep 'it' as temp_array[it,ix,iy].
        logging.debug('Transposing temperatures and multiplying by HbarC to convert fm^-1 to GeV')
        temp_data = self.temp_conv_factor * np.transpose(np.reshape(temp_data, [self.NT, self.grid_width, self.grid_width]), axes=[0, 2, 1])

        return temp_data

    # Method to get raw temp x-direction gradient data
    def temp_grad_x_array(self):
        # Get the temp data as an array organized to be [time, x, y]-ish
        temp_data = self.temp_array()

        # Compute temperature gradient in the x-direction
        temp_grad_x = np.gradient(temp_data, self.gridstep, axis=1)

        # Spit out the array, organized the same as the temp_data array
        return temp_grad_x

    # Method to get raw temp y-direction gradient data
    def temp_grad_y_array(self):
        # Get the temp data as an array organized to be [time, x, y]-ish
        temp_data = self.temp_array()

        # Compute temperature gradient in the y-direction
        temp_grad_y = np.gradient(temp_data, self.gridstep, axis=2)

        # Spit out the array, organized the same as the temp_data array
        return temp_grad_y

    # Method to get raw x velocity data
    def x_vel_array(self):
        # Cut x velocity data out and convert to numpy array
        x_vel_column = self.grid_data[['xvel']]
        x_vel_data = pd.DataFrame(x_vel_column).to_numpy()

        # Reshape this nonsense into an array with first index time, then x, then y.
        # You can get the x velocity at the grid point (ix,iy) at timestep it as vel_x_data[it,ix,iy].
        # The transposition has been confirmed against data.
        x_vel_data = np.transpose(np.reshape(x_vel_data, [self.NT, self.grid_width, self.grid_width]), axes=[0, 2, 1])

        return x_vel_data

    # Method to get raw y velocity data
    def y_vel_array(self):
        # Cut x velocity data out and convert to numpy array
        y_vel_column = self.grid_data[['yvel']]
        y_vel_data = pd.DataFrame(y_vel_column).to_numpy()

        # Reshape this nonsense into an array with first index time, then x, then y.
        # You can get the x velocity at the grid point (ix,iy) at timestep it as vel_x_data[it,ix,iy].
        # The transposition has been confirmed against data.
        y_vel_data = np.transpose(np.reshape(y_vel_data, [self.NT, self.grid_width, self.grid_width]), axes=[0, 2, 1])

        return y_vel_data

    # Method to get raw flow x-direction gradient data
    def grad_x_u_x_array(self):
        # Get the temp data as an array organized to be [time, x, y]-ish
        flow_x_data = self.x_vel_array()

        # Compute temperature gradient in the x-direction
        grad_x_u_x = np.gradient(flow_x_data, self.gridstep, axis=1)

        # Spit out the array, organized the same as the temp_data array
        return grad_x_u_x

    # Method to get raw flow x-direction gradient data
    def grad_x_u_y_array(self):
        # Get the temp data as an array organized to be [time, x, y]-ish
        flow_y_data = self.y_vel_array()

        # Compute temperature gradient in the x-direction
        grad_x_u_y = np.gradient(flow_y_data, self.gridstep, axis=1)

        # Spit out the array, organized the same as the temp_data array
        return grad_x_u_y

    # Method to get raw flow x-direction gradient data
    def grad_y_u_x_array(self):
        # Get the temp data as an array organized to be [time, x, y]-ish
        flow_x_data = self.x_vel_array()

        # Compute temperature gradient in the x-direction
        grad_y_u_x = np.gradient(flow_x_data, self.gridstep, axis=2)

        # Spit out the array, organized the same as the temp_data array
        return grad_y_u_x

    # Method to get raw flow x-direction gradient data
    def grad_y_u_y_array(self):
        # Get the temp data as an array organized to be [time, x, y]-ish
        flow_y_data = self.y_vel_array()

        # Compute temperature gradient in the x-direction
        grad_y_u_y = np.gradient(flow_y_data, self.gridstep, axis=2)

        # Spit out the array, organized the same as the temp_data array
        return grad_y_u_y

    # Method to plot raw temp data
    def plot_temps(self, time):
        # Cut temp data out and convert to numpy array #
        temp_data_cut = self.grid_data[['temp']]
        temp_data = pd.DataFrame(temp_data_cut).to_numpy()

        return plt.contourf(temp_data[time, :, :])

    # Method to return interpolated function object from data
    # Function to interpolate the temperature grid from the hydro file
    # Returns interpolating callable function
    def interpolate_temp_grid(self):
        print('Interpolating temp grid data for event: ' + str(self.name))

        # Cut temp data out and convert to numpy array
        temp_data = self.temp_array()

        # Interpolate data!
        # The final temperatures have been confirmed directly against absolute coordinates in data.
        interp_temps = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), temp_data)

        return interp_temps

    # Method to return interpolated function object from data
    # Function to interpolate the temperature gradient grid from the hydro file
    # Returns interpolating callable function
    def interpolate_temp_grad_x_grid(self):
        print('Interpolating temp x-gradient grid data for event: ' + str(self.name))

        # Cut temp data out and convert to numpy array
        temp_grad_x_data = self.temp_grad_x_array()

        # Interpolate data!
        # The final temperatures have been confirmed directly against absolute coordinates in data.
        interp_temp_grad_x = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), temp_grad_x_data)

        return interp_temp_grad_x

    # Method to return interpolated function object from data
    # Function to interpolate the temperature gradient grid from the hydro file
    # Returns interpolating callable function
    def interpolate_temp_grad_y_grid(self):
        print('Interpolating temp y-gradient grid data for event: ' + str(self.name))

        # Cut temp data out and convert to numpy array
        temp_grad_y_data = self.temp_grad_y_array()

        # Interpolate data!
        # The final temperatures have been confirmed directly against absolute coordinates in data.
        interp_temp_grad_y = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), temp_grad_y_data)

        return interp_temp_grad_y

    # Method to interpolate the x velocities from a grid file
    # Returns interpolating callable function
    def interpolate_x_vel_grid(self):
        print('Interpolating x-flow grid data for event: ' + str(self.name))

        # Cut x velocity data out and convert to numpy array
        x_vel_array = self.x_vel_array()

        # Interpolate data!
        # The final velocities have been confirmed directly against absolute coordinates in data.
        interp_x_vel = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), x_vel_array)

        return interp_x_vel

    # Method to interpolate the y velocities from a grid file
    # Returns interpolating callable function
    def interpolate_y_vel_grid(self):
        print('Interpolating y-flow grid data for event: ' + str(self.name))

        # Cut y velocity data out and convert to numpy array
        y_vel_array = self.y_vel_array()

        # Interpolate data!
        # The final velocities have been confirmed directly against absolute coordinates in data.
        interp_y_vel = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), y_vel_array)

        return interp_y_vel

    # Method to return interpolated function object from data
    # Function to interpolate the x direction flow gradient grid from the hydro file
    # Returns interpolating callable function
    def interpolate_grad_x_u_x_grid(self):
        print('Interpolating x-flow x-gradient grid data for event: ' + str(self.name))

        # Cut flow grad x data out and convert to numpy array
        grad_x_u_x_data = self.grad_x_u_x_array()

        # Interpolate data!
        interp_grad_x_u_x = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), grad_x_u_x_data)

        return interp_grad_x_u_x

    # Method to return interpolated function object from data
    # Function to interpolate the x direction flow gradient grid from the hydro file
    # Returns interpolating callable function
    def interpolate_grad_x_u_y_grid(self):
        print('Interpolating y-flow x-gradient grid data for event: ' + str(self.name))

        # Cut flow grad x data out and convert to numpy array
        grad_x_u_y_data = self.grad_x_u_y_array()

        # Interpolate data!
        interp_grad_x_u_y = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), grad_x_u_y_data)

        return interp_grad_x_u_y

    # Method to return interpolated function object from data
    # Function to interpolate the x direction flow gradient grid from the hydro file
    # Returns interpolating callable function
    def interpolate_grad_y_u_x_grid(self):
        print('Interpolating x-flow y-gradient grid data for event: ' + str(self.name))

        # Cut flow grad x data out and convert to numpy array
        grad_y_u_x_data = self.grad_y_u_x_array()

        # Interpolate data!
        interp_grad_y_u_x = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), grad_y_u_x_data)

        return interp_grad_y_u_x

    # Method to return interpolated function object from data
    # Function to interpolate the x direction flow gradient grid from the hydro file
    # Returns interpolating callable function
    def interpolate_grad_y_u_y_grid(self):
        print('Interpolating y-flow y-gradient grid data for event: ' + str(self.name))

        # Cut flow grad x data out and convert to numpy array
        grad_y_u_y_data = self.grad_y_u_y_array()

        # Interpolate data!
        interp_grad_y_u_y = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), grad_y_u_y_data)

        return interp_grad_y_u_y

    # Method to find the maximum temperature of a hydro file object
    def max_temp(self, time='i'):
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        else:
            pass

        # Calculate temperatures and take maximum.
        maxTemp = np.amax(self.temp_array()[time, :, :])

        return maxTemp

    # Method to find the minimum temperature of a hydro file object
    def min_temp(self, time='i'):
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        else:
            pass

        # Calculate temperatures and take maximum.
        minTemp = np.amin(self.temp_array()[time, :, :])

        return minTemp


# Plasma object as used for integration and muckery
class plasma_event:
    def __init__(self, temp_func=None, x_vel_func=None, y_vel_func=None, grad_x_func=None, grad_y_func=None,
                 grad_x_u_x_func=None, grad_x_u_y_func=None, grad_y_u_x_func=None, grad_y_u_y_func=None,
                 event=None, name=None, rmax=None):
        # Initialize all the ordinary plasma parameters
        if event is not None:
            self.temp = event.interpolate_temp_grid()
            self.x_vel = event.interpolate_x_vel_grid()
            self.y_vel = event.interpolate_y_vel_grid()
            self.temp_grad_x = event.interpolate_temp_grad_x_grid()
            self.temp_grad_y = event.interpolate_temp_grad_y_grid()
            self.grad_x_u_x = event.interpolate_grad_x_u_x_grid()
            self.grad_x_u_y = event.interpolate_grad_x_u_y_grid()
            self.grad_y_u_x = event.interpolate_grad_y_u_x_grid()
            self.grad_y_u_y = event.interpolate_grad_y_u_y_grid()
            self.name = event.name
            self.timestep = event.timestep
            self.t0 = np.amin(self.temp.grid[0])
            self.tf = np.amax(self.temp.grid[0])
            self.xmin = np.amin(self.temp.grid[1])
            self.xmax = np.amax(self.temp.grid[1])
            self.ymin = np.amin(self.temp.grid[2])
            self.ymax = np.amax(self.temp.grid[2])
            self.gridstep = event.gridstep
        elif temp_func is not None and x_vel_func is not None and y_vel_func is not None:
            self.temp = temp_func
            self.x_vel = x_vel_func
            self.y_vel = y_vel_func
            self.temp_grad_x = grad_x_func
            self.temp_grad_y = grad_y_func
            self.grad_x_u_x = grad_x_u_x_func
            self.grad_x_u_y = grad_x_u_y_func
            self.grad_y_u_x = grad_y_u_x_func
            self.grad_y_u_y = grad_y_u_y_func
            self.name = name
            try:
                # Attempt to get timestep as if the functions are regular interpolator objects.
                self.timestep = self.temp.grid[0][-1] - self.temp.grid[0][-2]
                self.t0 = np.amin(self.temp.grid[0])
                self.tf = np.amax(self.temp.grid[0])
                self.xmin = np.amin(self.temp.grid[1])
                self.xmax = np.amax(self.temp.grid[1])
                self.ymin = np.amin(self.temp.grid[2])
                self.ymax = np.amax(self.temp.grid[2])
            except AttributeError:
                # Set default values for parameters
                logging.warning('No valid parameters for event. Setting to defaults.')
                self.timestep = 0.1
                self.t0 = 0
                self.tf = 15
                self.xmin = -15
                self.xmax = 15
                self.ymin = -15
                self.ymax = 15
        else:
            print('Plasma instantiation failed.')
            raise Exception

        self.rmax = rmax



    # Method to get array on space domain of event with given resolution
    def xspace(self, resolution=100, fraction=1):
        return np.arange(start=fraction*self.xmin, stop=fraction*self.xmax,
                         step=((fraction*self.xmax - fraction*self.xmin) / resolution))

    # Method to get array on time domain of event with given resolution
    def tspace(self, resolution=100):
        return np.arange(start=self.t0, stop=self.tf, step=((self.tf - self.t0) / resolution))

    # Method to return the total magnitude of the velocity at a given point
    def vel(self, point=None):
        return np.sqrt(self.x_vel(point) ** 2 + self.y_vel(point) ** 2)

    # Method to return angle of velocity vector at a given point
    def vel_angle(self, point=None):
        current_point = point

        # np.arctan2 gives a signed angle, as opposed to np.arctan
        arctan2 = np.arctan2(self.y_vel(current_point), self.x_vel(current_point))

        # if the angle was negative, we need to correct it to return an angle on the domain [0, 2pi]
        if arctan2 < 0:
            return 2 * np.pi + arctan2  # Here we add the negative angle, reducing to corresponding value on [0, 2pi]
        else:
            return arctan2

    # Method to return velocity perpendicular to given trajectory angle at given time
    def u_perp(self, point, phi):
        return -self.x_vel(point) * np.sin(phi) \
               + self.y_vel(np.array(point)) * np.cos(phi)

    # Method to return velocity parallel to given trajectory angle at given time
    def u_par(self, point, phi):
        return self.x_vel(point) * np.cos(phi) \
               + self.y_vel(np.array(point)) * np.sin(phi)

    # Method to return gradient of the Temperature
    # at a particular point perpendicular to a given angle phi.
    # Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.
    def grad_perp_T(self, point, phi):
        # Compute x and y temperature gradient at given point, make grad vector
        grad_x = self.temp_grad_x(point)
        grad_y = self.temp_grad_y(point)
        grad_T = np.array([grad_x, grad_y])

        # Compute unit vector perpendicular to given phi
        e_perp = np.array([-np.sin(phi), np.cos(phi)])

        # Compute temperature gradient perp to given phi
        grad_perp_T = np.dot(e_perp, grad_T)  #(grad_x * e_perp[0]) + (grad_y * e_perp[1])

        return grad_perp_T

    # Method to return perp grad u perp, relative to given angle
    def grad_perp_u_perp(self, point, phi):
        # This is (eperp . grad) * (eperp . u)
        return (self.grad_x_u_x(point) * (np.sin(phi)**2)
                - self.grad_y_u_x(point) * np.sin(phi)*np.cos(phi)
                - self.grad_x_u_y(point) * np.sin(phi)*np.cos(phi)
                + self.grad_y_u_y(point) * (np.cos(phi)**2))

    # Method to return perp grad u par, relative to given angle
    def grad_perp_u_par(self, point, phi):
        # This is (eperp . grad) * (epar . u)
        return (- self.grad_x_u_x(point) * np.sin(phi) * np.cos(phi)
                + self.grad_y_u_x(point) * (np.cos(phi)**2)
                - self.grad_x_u_y(point) * (np.sin(phi)**2)
                + self.grad_y_u_y(point) * np.sin(phi) * np.cos(phi))

    # Method to return gradient of the flow
    # at a particular point parallel to a given angle phi.
    def grad_par_flow(self, point, phi):
        # Compute x and y temperature gradient at given point
        grad_x = self.flow_grad_x(point)
        grad_y = self.flow_grad_y(point)

        # Compute unit vector parallel to given phi
        e_perp = np.array([np.cos(phi), np.sin(phi)])

        # Compute temperature gradient perp to given phi
        grad_perp_flow = (grad_x * e_perp[0]) + (grad_y * e_perp[1])

        return grad_perp_flow

    # Method to find the maximum temperature of a plasma object
    def max_temp(self, resolution=100, time='i'):
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        else:
            pass

        # Adapted from grid_reader.qgp_plot()
        #
        # Domains of physical positions to plot at (in fm)
        # These limits of the linear space obtain the largest and smallest input value for
        # the interpolating function's position inputs.
        x_sample_space = self.xspace(resolution=resolution)

        # Create arrays of each coordinate
        # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
        # We necessarily must set time equal to a constant to plot in 2D.
        x_coords, y_coords = np.meshgrid(x_sample_space, x_sample_space, indexing='ij')
        # t_coords set to be an array matching the length of x_coords full of constant time
        # Note that we select "initial time" as 0.5 fs by default
        t_coords = np.full_like(x_coords, time)

        # Put coordinates together into ordered pairs.
        points = np.transpose(np.array([t_coords, x_coords, y_coords]), (2, 1, 0))

        # Calculate temperatures and take maximum.
        maxTemp = np.amax(self.temp(points))

        return maxTemp

    # Method to find the maximum temperature of a plasma object
    def min_temp(self, resolution=100, time='i'):
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        else:
            pass

        # Adapted from grid_reader.qgp_plot()
        #
        # Domains of physical positions to plot at (in fm)
        # These limits of the linear space obtain the largest and smallest input value for
        # the interpolating function's position inputs.
        x_sample_space = np.linspace(self.xmin, self.xmax, resolution)

        # Create arrays of each coordinate
        # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
        # We necessarily must set time equal to a constant to plot in 2D.
        x_coords, y_coords = np.meshgrid(x_sample_space, x_sample_space, indexing='ij')
        # t_coords set to be an array matching the length of x_coords full of constant time
        # Note that we select "initial time" as 0.5 fs by default
        t_coords = np.full_like(x_coords, time)

        # Put coordinates together into ordered pairs.
        points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

        # Calculate temperatures and take maximum.
        minTemp = np.amin(self.temp(points))

        return minTemp

    # Method to find the maximum temperature of a plasma object
    def temp_stats(self, resolution=100, time='i'):
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        else:
            pass

        # Adapted from grid_reader.qgp_plot()
        #
        # Domains of physical positions to plot at (in fm)
        # These limits of the linear space obtain the largest and smallest input value for
        # the interpolating function's position inputs.
        x_sample_space = self.xspace(resolution=resolution)

        # Create arrays of each coordinate
        # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
        # We necessarily must set time equal to a constant to plot in 2D.
        x_coords, y_coords = np.meshgrid(x_sample_space, x_sample_space, indexing='ij')
        # t_coords set to be an array matching the length of x_coords full of constant time
        # Note that we select "initial time" as 0.5 fs by default
        t_coords = np.full_like(x_coords, time)

        # Put coordinates together into ordered pairs.
        points = np.transpose(np.array([t_coords, x_coords, y_coords]), (2, 1, 0))

        # Calculate temperatures and take maximum.
        temp_threshold = 0.01  # threshold in GeV
        fluid_cell_temps = self.temp(points)  # Get array of temps
        fluid_cell_temps[(fluid_cell_temps < temp_threshold)] = np.nan  # Put all values < temp_threshold to np.nan
        maxTemp = np.nanmax(fluid_cell_temps)
        minTemp = np.nanmin(fluid_cell_temps)
        meanTemp = np.nanmean(fluid_cell_temps)
        medianTemp = np.nanmedian(fluid_cell_temps)
        stdTemp = np.nanstd(fluid_cell_temps)

        return maxTemp, minTemp, meanTemp, medianTemp, stdTemp

    # Method to find the maximum temperature of a plasma object
    def ext_vec_mag(self, vec='vel', ext='max', resolution=100, time='i'):
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        else:
            pass

        # Adapted from grid_reader.qgp_plot()
        #
        # Domains of physical positions to plot at (in fm)
        # These limits of the linear space obtain the largest and smallest input value for
        # the interpolating function's position inputs.
        x_sample_space = self.xspace(resolution=resolution)

        # Create arrays of each coordinate
        # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
        # We necessarily must set time equal to a constant to plot in 2D.
        x_coords, y_coords = np.meshgrid(x_sample_space, x_sample_space, indexing='ij')
        # t_coords set to be an array matching the length of x_coords full of constant time
        # Note that we select "initial time" as 0.5 fs by default
        t_coords = np.full_like(x_coords, time)

        # Put coordinates together into ordered pairs.
        points = np.transpose(np.array([t_coords, x_coords, y_coords]), (2, 1, 0))

        # Compute vectors at sample points
        if vec == 'vel':
            x_vec_mags = self.x_vel(points)
            y_vec_mags = self.y_vel(points)
        elif vec == 'grad':
            x_vec_mags = self.temp_grad_x(points)
            y_vec_mags = self.temp_grad_y(points)
        else:
            x_vec_mags = 0
            y_vec_mags = 0

        # Calculate magnitude
        vec_mags = np.sqrt(x_vec_mags ** 2 + y_vec_mags ** 2)

        # Determine extrema type and take extrema
        if ext == 'max':
            ext_vec_mag = np.amax(vec_mags)
        elif ext == 'min':
            ext_vec_mag = np.amin(vec_mags)
        else:
            ext_vec_mag = 0

        return ext_vec_mag

    # Method to plot interpolated temperature function and / or velocity field
    # Can plot contour or density / colormesh for temps, stream or quiver for velocities
    # Other options can adjust the output.
    # Returns the plot object to make integration elsewhere nicer.
    def plot(self, time, temp_resolution=100, vel_resolution=30, grad_resolution=30,
             temptype='contour', veltype='stream', gradtype='stream', plot_temp=True, plot_vel=True, plot_grad=False,
             numContours=15, zoom=1):
        tempMax = self.max_temp()

        # Domains of physical positions to plot at (in fm)
        # These limits of the linear space obtain the largest and smallest input value for
        # the interpolating function's position inputs.
        x_space = self.xspace(resolution=temp_resolution, fraction=zoom)

        transposeAxes = (2, 1, 0)

        if plot_temp:
            # Create arrays of each coordinate
            # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
            # We necessarily must set time equal to a constant to plot in 2D.
            x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')

            # t_coords set to be an array matching the length of x_coords full of constant time
            t_coords = np.full_like(x_coords, time)

            # Put coordinates together into an ordered pair.
            points = np.transpose(np.array([t_coords, x_coords, y_coords]), transposeAxes)

            # Calculate temperatures
            temp_points = self.temp(points)
        else:
            temp_points = 0

        if plot_vel:

            # Create arrays of each coordinate
            # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
            # We necessarily must set time equal to a constant to plot in 2D.
            x_space_vel = self.xspace(resolution=vel_resolution, fraction=zoom)
            vel_x_coords, vel_y_coords = np.meshgrid(x_space_vel, x_space_vel, indexing='ij')

            # t_coords set to be an array matching the length of x_coords full of constant time
            vel_t_coords = np.full_like(vel_x_coords, time)

            # t_coords set to be an array matching the length of x_coords full of constant time
            vel_points = np.transpose(np.array([vel_t_coords, vel_x_coords, vel_y_coords]), transposeAxes)

            # Calculate velocities
            x_vels = self.x_vel(vel_points)
            y_vels = self.y_vel(vel_points)

        else:
            x_vels = 0
            y_vels = 0

        if plot_grad:

            # Create arrays of each coordinate
            # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
            # We necessarily must set time equal to a constant to plot in 2D.
            x_space_grad = self.xspace(resolution=grad_resolution, fraction=zoom)
            grad_x_coords, grad_y_coords = np.meshgrid(x_space_grad, x_space_grad, indexing='ij')

            # t_coords set to be an array matching the length of x_coords full of constant time
            grad_t_coords = np.full_like(grad_x_coords, time)

            # t_coords set to be an array matching the length of x_coords full of constant time
            grad_points = np.transpose(np.array([grad_t_coords, grad_x_coords, grad_y_coords]), transposeAxes)

            # Calculate velocities
            grad_x = self.temp_grad_x(grad_points)
            grad_y = self.temp_grad_y(grad_points)

            # Find max and min grads
            grad_mags = np.sqrt(grad_x**2 + grad_y**2)
            grad_max = np.amax(grad_mags)
            grad_min = np.amin(grad_mags)

        else:
            grad_x = 0
            grad_y = 0


        # Make temperature plot
        if temptype == 'density' and plot_temp:
            temps = plt.pcolormesh(x_space, x_space, temp_points, cmap='plasma', shading='auto',
                                   norm=colors.Normalize(vmin=0, vmax=tempMax))
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            tempcb = plt.colorbar(temps, label='Temperature (GeV)',
                              format=tkr.FormatStrFormatter('%.2f'))
        elif temptype == 'contour' and plot_temp:
            tempLevels = np.linspace(0, tempMax, numContours)
            temps = plt.contourf(x_space, x_space, temp_points, cmap='plasma',
                                 norm=colors.Normalize(vmin=0, vmax=tempMax), levels=tempLevels)
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            tempcb = plt.colorbar(temps, label='Temperature (GeV)',
                              format=tkr.FormatStrFormatter('%.2f'))
        else:
            temps = 0
            tempcb = 0

        # Make velocity plot
        if veltype == 'stream' and plot_vel:
            vels = plt.streamplot(x_space_vel, x_space_vel, x_vels, y_vels,
                                  color=np.sqrt(x_vels ** 2 + y_vels ** 2),
                                  linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1))
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            velcb = plt.colorbar(vels.lines, label='Flow Velocity (c)')
        elif veltype == 'quiver' and plot_vel:
            vels = plt.quiver(x_space_vel, x_space_vel, x_vels, y_vels, np.sqrt(x_vels ** 2 + y_vels ** 2),
                              linewidth=1,
                              cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1))
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            velcb = plt.colorbar(vels, label='Flow Velocity (c)')
        else:
            vels = 0
            velcb = 0

        # Make gradient plot
        if gradtype == 'stream' and plot_grad:
            grads = plt.streamplot(x_space_grad, x_space_grad, grad_x, grad_y,
                                   color=np.sqrt(grad_x ** 2 + grad_y ** 2),
                                   linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=grad_max))
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            gradcb = plt.colorbar(grads.lines, label='Temp Grad (GeV / fm)')
        elif gradtype == 'quiver' and plot_grad:
            grads = plt.quiver(x_space_grad, x_space_grad, grad_x, grad_y, np.sqrt(grad_x ** 2 + grad_y ** 2),
                              linewidth=1,
                              cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=grad_max))
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            gradcb = plt.colorbar(grads, label='Temp Grad (GeV / fm)')
        else:
            grads = 0
            gradcb = 0

        return temps, vels, grads, tempcb, velcb, gradcb

# Takes callable functions that take parameters (t, x, y) for the temperature and velocities
# and returns plasma_event objects generated from them.
def tabulated_plasma(t_space, x_space, temp_values, x_vel_values, y_vel_values, name=None, return_grids=False):
    print('WARNING: Gradients of temp and flow not verified')
    grid_step = float(x_space[-1] - x_space[-2])
    rmax = x_space[-1]

    # Compute gradients
    temp_grad_x_values = np.gradient(temp_values, grid_step, axis=1)
    temp_grad_y_values = np.gradient(temp_values, grid_step, axis=2)
    grad_x_u_x = np.gradient(x_vel_values, grid_step, axis=1)
    grad_y_u_x = np.gradient(x_vel_values, grid_step, axis=2)
    grad_x_u_y = np.gradient(y_vel_values, grid_step, axis=1)
    grad_y_u_y = np.gradient(y_vel_values, grid_step, axis=2)

    # Interpolate functions
    interped_temp_function = RegularGridInterpolator((t_space, x_space, x_space), temp_values)

    interped_x_vel_function = RegularGridInterpolator((t_space, x_space, x_space), x_vel_values)
    interped_y_vel_function = RegularGridInterpolator((t_space, x_space, x_space), y_vel_values)

    interped_grad_x_function = RegularGridInterpolator((t_space, x_space, x_space), temp_grad_x_values)
    interped_grad_y_function = RegularGridInterpolator((t_space, x_space, x_space), temp_grad_y_values)

    interped_grad_x_u_x = RegularGridInterpolator((t_space, x_space, x_space), grad_x_u_x)
    interped_grad_y_u_x = RegularGridInterpolator((t_space, x_space, x_space), grad_y_u_x)

    interped_grad_x_u_y = RegularGridInterpolator((t_space, x_space, x_space), grad_x_u_y)
    interped_grad_y_u_y = RegularGridInterpolator((t_space, x_space, x_space), grad_y_u_y)

    # Create and return plasma object
    plasma_object = plasma_event(temp_func=interped_temp_function,
                                 x_vel_func=interped_x_vel_function,
                                 y_vel_func=interped_y_vel_function,
                                 grad_x_func=interped_grad_x_function,
                                 grad_y_func=interped_grad_y_function,
                                 grad_x_u_x_func=interped_grad_x_u_x,
                                 grad_y_u_x_func=interped_grad_y_u_x,
                                 grad_x_u_y_func=interped_grad_x_u_y,
                                 grad_y_u_y_func=interped_grad_y_u_y,
                                 name=name, rmax=rmax)

    # Return the grids of evaluated points, if requested.
    if return_grids:
        return plasma_object, temp_values, x_vel_values, y_vel_values,
    else:
        return plasma_object

# Takes callable functions that take parameters (t, x, y) for the temperature and velocities
# and returns plasma_event objects generated from them.
def functional_plasma(temp_func=None, x_vel_func=None, y_vel_func=None, name=None,
                      resolution=10, rmax=15, time=None, return_grids=False, tau0=0.5):
    # Define grid time and space domains
    if time is None:
        t_space = np.linspace(tau0, 2 * rmax, int((rmax + rmax) * resolution))
    else:
        t_space = np.linspace(tau0, time, int((rmax + rmax) * resolution))
    x_space = np.linspace((0 - rmax), rmax, int((rmax + rmax) * resolution))


    # Create meshgrid for function evaluation
    t_coords, x_coords, y_coords = np.meshgrid(t_space, x_space, x_space, indexing='ij')

    # Evaluate functions for grid points
    temp_values = temp_func(t_coords, x_coords, y_coords)
    x_vel_values = x_vel_func(t_coords, x_coords, y_coords)
    y_vel_values = y_vel_func(t_coords, x_coords, y_coords)

    # Create the plasma object via tabulated points
    return tabulated_plasma(t_space, x_space, temp_values, x_vel_values, y_vel_values, name=name,
                            return_grids=return_grids)




