"""
plasma.py – QGP medium representation for HIC Monte Carlo.

plasma_event stores raw hydro grid arrays and interpolates on the fly,
making it fully pickle-able (no lambdas / closures / interpolator objects
stored as attributes).

Public API:
  Fields (callable with pts of shape (...,4) = [tau, x, y, eta_s]):
    temp, x_vel, y_vel, z_vel,
    temp_grad_x, temp_grad_y, temp_grad_z,
    grad_x_u_x, grad_x_u_y, grad_x_u_z,
    grad_y_u_x, grad_y_u_y, grad_y_u_z,
    grad_z_u_x, grad_z_u_y, grad_z_u_z,
    vel

  Domain metadata:
    t0, tf, timestep, xmin, xmax, ymin, ymax, gridstep

  Methods:
    tspace(resolution), xspace(resolution, fraction),
    max_temp(...), min_temp(...), temp_stats(...), plot(...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

try:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tkr
except ImportError:
    print('NO MATPLOTLIB')

# class osu_hydro_file():

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_osu_hydro(file_path: str, temp_conv_factor: float = 0.1973269788):
    """
    Read an osu-hydro output file and return the raw grid arrays.

    Returns
    -------
    tspace : ndarray, shape (NT,)
    xspace : ndarray, shape (NX,)
    temp   : ndarray, shape (NT, NX, NX)   – in GeV
    ux     : ndarray, shape (NT, NX, NX)
    uy     : ndarray, shape (NT, NX, NX)
    """
    grid_data = pd.read_table(
        file_path, header=None, sep=r'\s+', dtype=np.float64,
        names=['time', 'xpos', 'ypos', 'temp', 'xvel', 'yvel'],
    )

    xlist = grid_data['xpos'].to_numpy()
    tlist = grid_data['time'].to_numpy()

    grid_width = int(np.sqrt(grid_data[['time']].value_counts().to_numpy()[-1]))
    n_grid_spaces = grid_width ** 2
    NT = int(len(tlist) / n_grid_spaces)

    timestep = tlist[-1] - tlist[-1 - n_grid_spaces]
    gridstep = np.abs(xlist[-1] - xlist[-2])

    tspace = np.linspace(np.amin(tlist) - timestep, np.amax(tlist), NT)
    xspace = np.linspace(np.amin(xlist), np.amax(xlist), grid_width)

    def _reshape(col):
        return np.transpose(
            np.reshape(grid_data[col].to_numpy(), [NT, grid_width, grid_width]),
            axes=[0, 2, 1],
        )

    temp = temp_conv_factor * _reshape('temp')
    ux   = _reshape('xvel')
    uy   = _reshape('yvel')

    return tspace, xspace, temp, ux, uy


def _interp(arr3d: np.ndarray, tspace: np.ndarray, xspace: np.ndarray,
            pts: np.ndarray) -> np.ndarray:
    """
    Build a temporary RegularGridInterpolator for *arr3d* and evaluate at *pts*.

    *pts* may have shape (..., 3) [tau, x, y] or (..., 4) [tau, x, y, eta_s].
    eta_s is silently dropped (boost-invariant approximation).
    """
    p = np.asarray(pts)
    if p.shape[-1] == 4:
        p = p[..., :3]
    elif p.shape[-1] != 3:
        raise ValueError(
            f"Expected points with last dim 3 or 4, got shape {pts.shape}"
        )
    interp = RegularGridInterpolator(
        (tspace, xspace, xspace), arr3d, bounds_error=False, fill_value=None
    )
    return interp(p)


# ---------------------------------------------------------------------------
# plasma_event
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class plasma_event:
    """
    Medium fields in Milne coordinates (tau, x, y, eta_s) assuming
    longitudinal boost invariance (2+1D hydro).

    All field methods accept points of shape (..., 4) = [tau, x, y, eta_s].

    The object is fully pickle-able: only raw numpy arrays and plain scalars
    are stored as attributes.

    Construction
    ------------
    From a file:
        plasma_event(hydro_file_path="path/to/evo.dat")

    From pre-computed arrays (used by tabulated_plasma / functional_plasma):
        plasma_event(
            tspace=..., xspace=...,
            _T=..., _ux=..., _uy=...,
        )
    """

    # --- init inputs -------------------------------------------------------
    hydro_file_path: Optional[str] = None
    name: Optional[str] = None
    meta: Optional[dict] = None
    temp_conv_factor: float = 0.1973269788  # fm^-1 → GeV

    # Raw arrays (may be supplied directly or filled in __post_init__)
    _tspace: Optional[np.ndarray] = field(default=None, repr=False)
    _xspace: Optional[np.ndarray] = field(default=None, repr=False)
    _T:      Optional[np.ndarray] = field(default=None, repr=False)  # (NT,NX,NX)
    _ux:     Optional[np.ndarray] = field(default=None, repr=False)
    _uy:     Optional[np.ndarray] = field(default=None, repr=False)

    # Gradient arrays (computed in __post_init__)
    _dT_dx:    Optional[np.ndarray] = field(default=None, repr=False)
    _dT_dy:    Optional[np.ndarray] = field(default=None, repr=False)
    _dux_dx:   Optional[np.ndarray] = field(default=None, repr=False)
    _dux_dy:   Optional[np.ndarray] = field(default=None, repr=False)
    _duy_dx:   Optional[np.ndarray] = field(default=None, repr=False)
    _duy_dy:   Optional[np.ndarray] = field(default=None, repr=False)

    # --- domain metadata (set in __post_init__) ----------------------------
    timestep: float = field(init=False)
    t0:       float = field(init=False)
    tf:       float = field(init=False)
    xmin:     float = field(init=False)
    xmax:     float = field(init=False)
    ymin:     float = field(init=False)
    ymax:     float = field(init=False)
    gridstep: float = field(init=False)

    # -----------------------------------------------------------------------

    def __post_init__(self) -> None:
        # --- 1. Source the raw arrays ---------------------------------------
        if self.hydro_file_path is not None:
            logging.info(f"Reading osu-hydro file: {self.hydro_file_path}")
            tsp, xsp, T, ux, uy = _read_osu_hydro(
                self.hydro_file_path, self.temp_conv_factor
            )
            self._tspace = tsp
            self._xspace = xsp
            self._T  = T
            self._ux = ux
            self._uy = uy

        if self._T is None or self._ux is None or self._uy is None:
            raise ValueError(
                "plasma_event: supply hydro_file_path OR (_tspace, _xspace, _T, _ux, _uy)."
            )

        # --- 2. Pre-compute spatial gradients (once, from raw arrays) -------
        gs = float(self._xspace[-1] - self._xspace[-2])  # gridstep
        if self._dT_dx is None:
            self._dT_dx  = np.gradient(self._T,  gs, axis=1)
        if self._dT_dy is None:
            self._dT_dy  = np.gradient(self._T,  gs, axis=2)
        if self._dux_dx is None:
            self._dux_dx = np.gradient(self._ux, gs, axis=1)
        if self._dux_dy is None:
            self._dux_dy = np.gradient(self._ux, gs, axis=2)
        if self._duy_dx is None:
            self._duy_dx = np.gradient(self._uy, gs, axis=1)
        if self._duy_dy is None:
            self._duy_dy = np.gradient(self._uy, gs, axis=2)

        # --- 3. Domain metadata --------------------------------------------
        self.t0       = float(np.amin(self._tspace))
        self.tf       = float(np.amax(self._tspace))
        self.timestep = float(self._tspace[-1] - self._tspace[-2])
        self.xmin     = float(np.amin(self._xspace))
        self.xmax     = float(np.amax(self._xspace))
        self.ymin     = self.xmin
        self.ymax     = self.xmax
        self.gridstep = gs

    # -----------------------------------------------------------------------
    # Private interpolation helper (uses stored arrays; no closures)
    # -----------------------------------------------------------------------

    def _eval(self, arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Interpolate *arr* (shape NT×NX×NX) at *pts* (...,3 or 4)."""
        return _interp(arr, self._tspace, self._xspace, pts)

    # -----------------------------------------------------------------------
    # Field callables (preserve original names exactly)
    # -----------------------------------------------------------------------

    def temp(self, pts) -> np.ndarray:
        """Temperature [GeV] at pts = (..., [tau, x, y, eta_s])."""
        return self._eval(self._T, pts)

    def x_vel(self, pts) -> np.ndarray:
        """Lab-frame x-velocity = ux / cosh(eta_s)."""
        p = np.asarray(pts)
        ux = self._eval(self._ux, p)
        eta_s = p[..., 3] if p.shape[-1] == 4 else 0.0
        return ux / np.cosh(eta_s)

    def y_vel(self, pts) -> np.ndarray:
        """Lab-frame y-velocity = uy / cosh(eta_s)."""
        p = np.asarray(pts)
        uy = self._eval(self._uy, p)
        eta_s = p[..., 3] if p.shape[-1] == 4 else 0.0
        return uy / np.cosh(eta_s)

    def z_vel(self, pts) -> np.ndarray:
        """Lab-frame z-velocity = tanh(eta_s)."""
        p = np.asarray(pts)
        eta_s = p[..., 3] if p.shape[-1] == 4 else 0.0
        return np.tanh(eta_s)

    def vel(self, pts) -> np.ndarray:
        """Total velocity magnitude sqrt(vx²+vy²+vz²)."""
        return np.sqrt(self.x_vel(pts)**2 + self.y_vel(pts)**2 + self.z_vel(pts)**2)

    # Temperature gradients
    def temp_grad_x(self, pts) -> np.ndarray:
        return self._eval(self._dT_dx, pts)

    def temp_grad_y(self, pts) -> np.ndarray:
        return self._eval(self._dT_dy, pts)

    def temp_grad_z(self, pts) -> np.ndarray:
        """ Boost-invariant model: literal boost invariance of temperature """
        p = np.asarray(pts)
        return np.zeros(p.shape[:-1])

    # Flow velocity gradient tensor (spatial components)
    def grad_x_u_x(self, pts) -> np.ndarray:
        return self._eval(self._dux_dx, pts)

    def grad_x_u_y(self, pts) -> np.ndarray:
        return self._eval(self._duy_dx, pts)

    def grad_x_u_z(self, pts) -> np.ndarray:
        """ Boost-invariant model: z velocity is only dependent on eta_s """
        p = np.asarray(pts)
        return np.zeros(p.shape[:-1])

    def grad_y_u_x(self, pts) -> np.ndarray:
        return self._eval(self._dux_dy, pts)

    def grad_y_u_y(self, pts) -> np.ndarray:
        return self._eval(self._duy_dy, pts)

    def grad_y_u_z(self, pts) -> np.ndarray:
        """ Boost-invariant model: z velocity is only dependent on eta_s """
        p = np.asarray(pts)
        return np.zeros(p.shape[:-1])

    def grad_z_u_x(self, pts) -> np.ndarray:
        """ Boost-invariant model: transverse velocity is z-independent """
        p = np.asarray(pts)
        return np.zeros(p.shape[:-1])

    def grad_z_u_y(self, pts) -> np.ndarray:
        """ Boost-invariant model: transverse velocity is z-independent """
        p = np.asarray(pts)
        return np.zeros(p.shape[:-1])

    def grad_z_u_z(self, pts) -> np.ndarray:
        """ 1 / (tau * cosh(eta_s)) — kinematic term from boost invariance. """
        p = np.asarray(pts)
        tau   = p[..., 0]
        eta_s = p[..., 3] if p.shape[-1] == 4 else 0.0
        return 1.0 / (tau * np.cosh(eta_s))

    # -----------------------------------------------------------------------
    # Domain helpers
    # -----------------------------------------------------------------------

    def xspace(self, resolution: int = 100, fraction: float = 1.0) -> np.ndarray:
        """Uniform array over [fraction·xmin, fraction·xmax] with *resolution* points."""
        lo = fraction * self.xmin
        hi = fraction * self.xmax
        return np.arange(lo, hi, (hi - lo) / resolution)

    def tspace(self, resolution: int = 100) -> np.ndarray:
        """Uniform array over [t0, tf] with *resolution* points."""
        return np.arange(self.t0, self.tf, (self.tf - self.t0) / resolution)

    # -----------------------------------------------------------------------
    # Temperature statistics
    # -----------------------------------------------------------------------

    def _grid_points(self, time: float, resolution: int) -> np.ndarray:
        """Return a (resolution, resolution, 3) array of (tau, x, y) grid points."""
        x_sp = self.xspace(resolution=resolution)
        x_coords, y_coords = np.meshgrid(x_sp, x_sp, indexing='ij')
        t_coords = np.full_like(x_coords, time)
        return np.transpose(np.array([t_coords, x_coords, y_coords]), (2, 1, 0))

    def max_temp(self, resolution: int = 100, time='i') -> float:
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        return float(np.amax(self.temp(self._grid_points(time, resolution))))

    def min_temp(self, resolution: int = 100, time='i') -> float:
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        return float(np.amin(self.temp(self._grid_points(time, resolution))))

    def temp_stats(self, resolution: int = 100, time='i'):
        if time == 'i':
            time = self.t0
        elif time == 'f':
            time = self.tf
        pts = self._grid_points(time, resolution)
        vals = self.temp(pts).astype(float)
        threshold = 0.01  # GeV
        vals[vals < threshold] = np.nan
        return (
            float(np.nanmax(vals)),
            float(np.nanmin(vals)),
            float(np.nanmean(vals)),
            float(np.nanmedian(vals)),
            float(np.nanstd(vals)),
        )

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot(self, time=None, temp_resolution=100, vel_resolution=100,
             grad_resolution=100, temptype='contour', veltype='stream',
             gradtype='stream', plot_temp=True, plot_vel=True, plot_grad=False,
             numContours=15, zoom=1, eta_s=0):
        if time is None:
            time = self.t0

        tempMax = self.max_temp()
        transposeAxes = (2, 1, 0)

        # --- Temperature ---
        if plot_temp:
            x_space = self.xspace(resolution=temp_resolution, fraction=zoom)
            x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
            t_coords = np.full_like(x_coords, time)
            points = np.transpose(np.array([t_coords, x_coords, y_coords]), transposeAxes)
            temp_points = self.temp(points)
        else:
            x_space = self.xspace(resolution=temp_resolution, fraction=zoom)
            temp_points = 0

        # --- Velocities ---
        if plot_vel:
            x_space_vel = self.xspace(resolution=vel_resolution, fraction=zoom)
            vel_x_coords, vel_y_coords = np.meshgrid(x_space_vel, x_space_vel, indexing='ij')
            vel_etas = np.full_like(vel_x_coords, eta_s)
            vel_t = np.full_like(vel_x_coords, time)
            vel_points = np.transpose(
                np.array([vel_t, vel_x_coords, vel_y_coords, vel_etas]), transposeAxes
            )
            x_vels = self.x_vel(vel_points)
            y_vels = self.y_vel(vel_points)
        else:
            x_space_vel = self.xspace(resolution=vel_resolution, fraction=zoom)
            x_vels = y_vels = 0

        # --- Gradients ---
        if plot_grad:
            x_space_grad = self.xspace(resolution=grad_resolution, fraction=zoom)
            gx_coords, gy_coords = np.meshgrid(x_space_grad, x_space_grad, indexing='ij')
            g_etas = np.full_like(gx_coords, eta_s)
            g_t = np.full_like(gx_coords, time)
            grad_points = np.transpose(
                np.array([g_t, gx_coords, gy_coords, g_etas]), transposeAxes
            )
            grad_x = self.temp_grad_x(grad_points)
            grad_y = self.temp_grad_y(grad_points)
            grad_mags = np.sqrt(grad_x**2 + grad_y**2)
            grad_max = float(np.amax(grad_mags))
        else:
            x_space_grad = self.xspace(resolution=grad_resolution, fraction=zoom)
            grad_x = grad_y = 0
            grad_max = 1.0

        # --- Render temperature ---
        if temptype == 'density' and plot_temp:
            temps = plt.pcolormesh(
                x_space, x_space, temp_points, cmap='plasma', shading='auto',
                norm=colors.Normalize(vmin=0, vmax=tempMax),
            )
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            tempcb = plt.colorbar(temps, label='Temperature (GeV)',
                                  format=tkr.FormatStrFormatter('%.2f'))
        elif temptype == 'contour' and plot_temp:
            tempLevels = np.linspace(0, tempMax, numContours)
            temps = plt.contourf(
                x_space, x_space, temp_points, cmap='plasma',
                norm=colors.Normalize(vmin=0, vmax=tempMax), levels=tempLevels,
            )
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            tempcb = plt.colorbar(temps, label='Temperature (GeV)',
                                  format=tkr.FormatStrFormatter('%.2f'))
        else:
            temps = tempcb = 0

        # --- Render velocities ---
        if veltype == 'stream' and plot_vel:
            vels = plt.streamplot(
                x_space_vel, x_space_vel, x_vels, y_vels,
                color=np.sqrt(x_vels**2 + y_vels**2),
                linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1),
            )
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            velcb = plt.colorbar(vels.lines, label='Flow Velocity (c)')
        elif veltype == 'quiver' and plot_vel:
            vels = plt.quiver(
                x_space_vel, x_space_vel, x_vels, y_vels,
                np.sqrt(x_vels**2 + y_vels**2),
                linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1),
            )
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            velcb = plt.colorbar(vels, label='Flow Velocity (c)')
        else:
            vels = velcb = 0

        # --- Render gradients ---
        if gradtype == 'stream' and plot_grad:
            grads = plt.streamplot(
                x_space_grad, x_space_grad, grad_x, grad_y,
                color=np.sqrt(grad_x**2 + grad_y**2),
                linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=grad_max),
            )
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            gradcb = plt.colorbar(grads.lines, label='Temp Grad (GeV / fm)')
        elif gradtype == 'quiver' and plot_grad:
            grads = plt.quiver(
                x_space_grad, x_space_grad, grad_x, grad_y,
                np.sqrt(grad_x**2 + grad_y**2),
                linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=grad_max),
            )
            plt.gca().set_aspect('equal')
            plt.gca().set_xlabel('X Position [fm]')
            plt.gca().set_ylabel('Y Position [fm]')
            gradcb = plt.colorbar(grads, label='Temp Grad (GeV / fm)')
        else:
            grads = gradcb = 0

        return temps, vels, grads, tempcb, velcb, gradcb


# Takes tabulated data for the temperature and velocities
# and returns plasma_event objects generated from them.
def tabulated_plasma(t_space, x_space, temp_values, x_vel_values, y_vel_values, name=None, return_grids=False):
    # Feed tabulated data to constructor -- plasma_event handles gradients
    plasma_object = plasma_event(
        name=name,
        _tspace=np.asarray(t_space),
        _xspace=np.asarray(x_space),
        _T=np.asarray(temp_values),
        _ux=np.asarray(x_vel_values),
        _uy=np.asarray(y_vel_values),
    )
    if return_grids:
        return plasma_object, temp_values, x_vel_values, y_vel_values
    return plasma_object


def functional_plasma(temp_func=None, x_vel_func=None, y_vel_func=None, name=None,
                      resolution=10, xmax=15, time=None, return_grids=False, tau0=0.5):
    """Build a plasma_event by evaluating callable functions on a grid."""
    if time is None:
        t_space = np.linspace(tau0, 2 * xmax, int((xmax + xmax) * resolution))
    else:
        t_space = np.linspace(tau0, time, int((xmax + xmax) * resolution))
    x_space = np.linspace(-xmax, xmax, int((xmax + xmax) * resolution))

    # Create meshgrid and evaluate functions
    t_coords, x_coords, y_coords = np.meshgrid(t_space, x_space, x_space, indexing='ij')
    temp_values  = temp_func(t_coords, x_coords, y_coords)
    x_vel_values = x_vel_func(t_coords, x_coords, y_coords)
    y_vel_values = y_vel_func(t_coords, x_coords, y_coords)

    # Give to tabulated plasma
    return tabulated_plasma(t_space, x_space, temp_values, x_vel_values, y_vel_values,
                            name=name, return_grids=return_grids)