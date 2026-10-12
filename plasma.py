"""
plasma.py – QGP medium representation for HIC Monte Carlo.

plasma_event stores raw hydro grid arrays and interpolates on the fly,
making it fully pickle-able (no lambdas / closures / interpolator objects
stored as attributes).

Supports two modes, selected at construction time via boost_invariant=:

  boost_invariant=True  (default)
    2+1D hydro output on a (tau, x, y) grid.
    eta_s is accepted in all call signatures but dropped internally.
    Boost-invariant kinematics are applied analytically:
        vz = tanh(eta_s),   grad_z_u_z = 1/(tau*cosh(eta_s))

  boost_invariant=False
    3+1D hydro output on a (tau, x, y, eta_s) grid.
    All four coordinates are used for interpolation.
    vz and all eta_s-gradient components are read directly from data.

Public API (identical for both modes):
  Field methods (pts shape (..., 4) = [tau, x, y, eta_s]):
    temp, x_vel, y_vel, z_vel,
    temp_grad_x, temp_grad_y, temp_grad_z,
    grad_x_u_x, grad_x_u_y, grad_x_u_z,
    grad_y_u_x, grad_y_u_y, grad_y_u_z,
    grad_z_u_x, grad_z_u_y, grad_z_u_z,
    vel

  Domain metadata:
    t0, tf, timestep, xmin, xmax, ymin, ymax, gridstep
    etasmin, etasmax  (3+1D only; set to 0.0/0.0 in 2+1D mode)

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

import config

try:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tkr
except ImportError:
    print('NO MATPLOTLIB')

##############################
# Equation of State Business #
##############################

# Function to return partial density at a particular point for given medium partons
# Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.
def rho(temp, soft_pid=21):
    if soft_pid == 21:
        density = 1.202056903159594 * 16 * (1 / (np.pi ** 2)) * temp ** 3
    elif np.abs(soft_pid) < 7:
        NDOF = 6  # 24 / 4  --> per pdg id, instead of for u, ubar, d, dbar together.
        density = 1.202056903159594 * (3/4) * NDOF * (1 / (np.pi ** 2)) * temp ** 3
    elif soft_pid == 0:  # Use all light quarks, plus gluons
        NDOF = 24  # for u, ubar, d, dbar together.
        q_density = 1.202056903159594 * (3 / 4) * NDOF * (1 / (np.pi ** 2)) * temp ** 3
        g_density = 1.202056903159594 * 16 * (1 / (np.pi ** 2)) * temp ** 3
        density = q_density + g_density
    else:
        # Return 0
        density = 0
    return density


# Function to return gradient of partial density at a particular point for given medium partons
# Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.
def gradrho(temp, gradtemp, soft_pid=21):
    if soft_pid == 21:
        graddensity = 1.202056903159594 * 16 * (1 / (np.pi ** 2)) * 3 * (temp ** 2) * gradtemp
    elif np.abs(soft_pid) < 7:
        NDOF = 6  # 24 / 4  --> per pdg id, instead of for u, ubar, d, dbar together.
        graddensity = 1.202056903159594 * (3/4) * NDOF * (1 / (np.pi ** 2)) * 3 * (temp ** 2) * gradtemp
    else:
        # Return 0
        graddensity = 0
    return graddensity


###################################
# Freeze-out hypersurface helpers #
###################################

def _find_temperature_crossing(T_grid: np.ndarray, tspace: np.ndarray,
                               T_switch: float=config.soft_transport.hydro.T_SWITCH):
    """
    For every spatial grid point, locate the proper time at which the
    temperature cools through T_switch, scanning along axis 0 (tau).

    Parameters
    ----------
    T_grid   : ndarray, shape (NT, ...)   temperature at each grid point
    tspace   : ndarray, shape (NT,)       proper-time grid
    T_switch : float                       switching temperature [GeV]

    Returns
    -------
    tau_fo : ndarray, shape (...)      interpolated crossing proper time (NaN where invalid)
    i_lo   : ndarray[int], shape (...) bracket index just above T_switch
    frac   : ndarray, shape (...)      fractional distance between i_lo and i_lo+1
    valid  : ndarray[bool], shape (...) True where a genuine cooling-through crossing exists
    """
    below = T_grid <= T_switch                       # (NT, ...)
    any_below = below.any(axis=0)
    first_below = np.argmax(below, axis=0)            # 0 if never below (filtered below)
    started_above = ~below[0]                         # must start hot

    valid = any_below & started_above & (first_below > 0)

    i_hi = np.clip(first_below, 1, len(tspace) - 1)
    i_lo = i_hi - 1

    T_lo = np.take_along_axis(T_grid, i_lo[None, ...], axis=0)[0]
    T_hi = np.take_along_axis(T_grid, i_hi[None, ...], axis=0)[0]

    denom = T_hi - T_lo
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(denom != 0, (T_switch - T_lo) / denom, 0.0)
    frac = np.clip(frac, 0.0, 1.0)

    tau_lo = tspace[i_lo]
    tau_hi = tspace[i_hi]
    tau_fo = tau_lo + frac * (tau_hi - tau_lo)
    tau_fo = np.where(valid, tau_fo, np.nan)

    return tau_fo, i_lo, frac, valid


def _interp_at_crossing(arr: np.ndarray, i_lo: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Linearly interpolate arr (shape (NT, ...)) between (i_lo, i_lo+1) using frac."""
    v_lo = np.take_along_axis(arr, i_lo[None, ...], axis=0)[0]
    v_hi = np.take_along_axis(arr, (i_lo + 1)[None, ...], axis=0)[0]
    return v_lo + frac * (v_hi - v_lo)


def _sample_fd_momentum(T: float, n: int, rng: np.random.Generator,
                         max_iter: int = 200) -> np.ndarray:
    """
    Sample n momenta from the massless Fermi-Dirac distribution
        f(p) dp ∝ p^2 / (exp(p/T) + 1) dp
    via rejection sampling against the Maxwell-Boltzmann envelope
        g(p) dp ∝ p^2 exp(-p/T) dp    (a Gamma(shape=3, scale=T) density),
    for which f(p)/g(p) = 1/(1+exp(-p/T)) in [0.5, 1], so acceptance = f/g.
    """
    out = np.empty(n, dtype=float)
    filled = 0
    it = 0
    while filled < n and it < max_iter:
        it += 1
        need = n - filled
        trial = rng.gamma(shape=3.0, scale=T, size=int(need * 1.5) + 8)
        accept_prob = 1.0 / (1.0 + np.exp(-trial / T))
        accepted = trial[rng.random(trial.shape[0]) < accept_prob]
        take = min(len(accepted), need)
        out[filled:filled + take] = accepted[:take]
        filled += take
    if filled < n:
        raise RuntimeError("Fermi-Dirac momentum sampling failed to converge.")
    return out


def _sample_be_momentum(T: float, n: int, rng: np.random.Generator,
                         n_max: int = 40) -> np.ndarray:
    """
    Sample n momenta from the massless Bose-Einstein distribution
        f(p) dp ∝ p^2 / (exp(p/T) - 1) dp
    using the exact series expansion 1/(e^x - 1) = sum_{k=1..inf} e^{-kx},
    which makes f(p) a (truncated) mixture of Gamma(shape=3, scale=T/k)
    densities with weights ∝ 1/k^3 (since ∫ p^2 e^{-kp/T} dp ∝ (T/k)^3).
    """
    k = np.arange(1, n_max + 1)
    weights = 1.0 / k**3
    weights /= weights.sum()
    k_sample = rng.choice(k, size=n, p=weights)
    return rng.gamma(shape=3.0, scale=T / k_sample)


#########################
# Internal Data Helpers #
#########################

def _read_osu_hydro(file_path: str, temp_conv_factor: float = 0.1973269788):
    """
    Read an osu-hydro (2+1D) output file and return the raw grid arrays.

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


def _make_interp_3d(arr3d, tspace, xspace):
    """Build a 3D RegularGridInterpolator on (tau, x, y)."""
    return RegularGridInterpolator(
        (tspace, xspace, xspace), arr3d, bounds_error=False, fill_value=None
    )


def _make_interp_4d(arr4d, tspace, xspace, etas_space):
    """Build a 4D RegularGridInterpolator on (tau, x, y, eta_s)."""
    return RegularGridInterpolator(
        (tspace, xspace, xspace, etas_space), arr4d, bounds_error=False, fill_value=None
    )


def _eval_3d(arr3d: np.ndarray, tspace: np.ndarray, xspace: np.ndarray,
             pts: np.ndarray) -> np.ndarray:
    """
    Interpolate a (NT, NX, NY) array at pts of shape (..., 3 or 4).
    eta_s (column 3) is silently dropped.
    """
    p = np.asarray(pts, dtype=float)
    if p.shape[-1] == 4:
        p = p[..., :3]
    elif p.shape[-1] != 3:
        raise ValueError(f"Expected last dim 3 or 4, got shape {pts.shape}")
    return _make_interp_3d(arr3d, tspace, xspace)(p)


def _eval_4d(arr4d: np.ndarray, tspace: np.ndarray, xspace: np.ndarray,
             etas_space: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Interpolate a (NT, NX, NY, NETA) array at pts of shape (..., 4).
    """
    p = np.asarray(pts, dtype=float)
    if p.shape[-1] != 4:
        raise ValueError(
            f"3+1D plasma requires points with last dim 4 [tau,x,y,eta_s], got {pts.shape}"
        )
    return _make_interp_4d(arr4d, tspace, xspace, etas_space)(p)


################
# plasma_event #
################

@dataclass(slots=True)
class plasma:
    """
    Medium fields in Milne coordinates (tau, x, y, eta_s).

    Set boost_invariant=True  for 2+1D hydro (osu-hydro style).
    Set boost_invariant=False for 3+1D hydro (full eta_s dependence).

    All field methods accept points of shape (..., 4) = [tau, x, y, eta_s].

    The object is fully pickle-able: only raw numpy arrays and plain scalars
    are stored as attributes.

    Construction
    ------------
    2+1D from a file:
        plasma_event(hydro_file_path="path/to/evo.dat")

    2+1D from pre-computed arrays:
        plasma_event(_tspace=..., _xspace=..., _T=..., _ux=..., _uy=...)

    3+1D from pre-computed arrays:
        plasma_event(
            boost_invariant=False,
            _tspace=..., _xspace=..., _etas_space=...,
            _T=..., _ux=..., _uy=..., _uz=...,
        )
    """

    # mode flag (must be set before arrays)
    boost_invariant: bool = True

    # init inputs
    hydro_file_path: Optional[str] = None
    name: Optional[str] = None
    meta: Optional[dict] = None
    temp_conv_factor: float = 0.1973269788  # fm^-1 → GeV

    # Grid axes (filled by __post_init__ if not supplied directly)
    _tspace:    Optional[np.ndarray] = field(default=None, repr=False)
    _xspace:    Optional[np.ndarray] = field(default=None, repr=False)
    _etas_space: Optional[np.ndarray] = field(default=None, repr=False)  # 3+1D only

    # Primary field arrays
    # 2+1D: shape (NT, NX, NX)
    # 3+1D: shape (NT, NX, NX, NETA)
    _T:  Optional[np.ndarray] = field(default=None, repr=False)
    _ux: Optional[np.ndarray] = field(default=None, repr=False)
    _uy: Optional[np.ndarray] = field(default=None, repr=False)
    _uz: Optional[np.ndarray] = field(default=None, repr=False)  # 3+1D only

    # Gradient arrays (computed in __post_init__ if not supplied)
    _dT_dx:    Optional[np.ndarray] = field(default=None, repr=False)
    _dT_dy:    Optional[np.ndarray] = field(default=None, repr=False)
    _dT_deta:  Optional[np.ndarray] = field(default=None, repr=False)  # 3+1D only
    _dux_dx:   Optional[np.ndarray] = field(default=None, repr=False)
    _dux_dy:   Optional[np.ndarray] = field(default=None, repr=False)
    _dux_deta: Optional[np.ndarray] = field(default=None, repr=False)  # 3+1D only
    _duy_dx:   Optional[np.ndarray] = field(default=None, repr=False)
    _duy_dy:   Optional[np.ndarray] = field(default=None, repr=False)
    _duy_deta: Optional[np.ndarray] = field(default=None, repr=False)  # 3+1D only
    _duz_dx:   Optional[np.ndarray] = field(default=None, repr=False)  # 3+1D only
    _duz_dy:   Optional[np.ndarray] = field(default=None, repr=False)  # 3+1D only
    _duz_deta: Optional[np.ndarray] = field(default=None, repr=False)  # 3+1D only

    # domain metadata (set in __post_init__)
    timestep: float = field(init=False)
    t0:       float = field(init=False)
    tf:       float = field(init=False)
    xmin:     float = field(init=False)
    xmax:     float = field(init=False)
    ymin:     float = field(init=False)
    ymax:     float = field(init=False)
    gridstep: float = field(init=False)
    etasmin:  float = field(init=False)  # 0.0 in boost-invariant mode
    etasmax:  float = field(init=False)  # 0.0 in boost-invariant mode

    # freeze-out hypersurface cache (populated lazily; plain dict of
    # picklable FreezeoutSurface objects, keyed by (T_switch, soft_pid))
    _freezeout_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # 1. Source the raw arrays
        if self.hydro_file_path is not None:
            if not self.boost_invariant:
                raise ValueError(
                    "hydro_file_path reader is for osu-hydro (2+1D) only. "
                    "Supply arrays directly for 3+1D mode."
                )
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

        if not self.boost_invariant:
            if self._etas_space is None or self._uz is None:
                raise ValueError(
                    "3+1D mode requires _etas_space and _uz arrays."
                )

        # 2. Pre-compute spatial gradients (once, from raw arrays)
        gs = float(self._xspace[-1] - self._xspace[-2])  # spatial grid step

        if self.boost_invariant:
            # 3D gradients only (axes: 0=tau, 1=x, 2=y)
            if self._dT_dx  is None: self._dT_dx  = np.gradient(self._T,  gs, axis=1)
            if self._dT_dy  is None: self._dT_dy  = np.gradient(self._T,  gs, axis=2)
            if self._dux_dx is None: self._dux_dx = np.gradient(self._ux, gs, axis=1)
            if self._dux_dy is None: self._dux_dy = np.gradient(self._ux, gs, axis=2)
            if self._duy_dx is None: self._duy_dx = np.gradient(self._uy, gs, axis=1)
            if self._duy_dy is None: self._duy_dy = np.gradient(self._uy, gs, axis=2)
        else:
            # 4D gradients (axes: 0=tau, 1=x, 2=y, 3=eta_s)
            es = float(self._etas_space[-1] - self._etas_space[-2])  # eta_s step
            if self._dT_dx   is None: self._dT_dx   = np.gradient(self._T,  gs, axis=1)
            if self._dT_dy   is None: self._dT_dy   = np.gradient(self._T,  gs, axis=2)
            if self._dT_deta is None: self._dT_deta = np.gradient(self._T,  es, axis=3)
            if self._dux_dx   is None: self._dux_dx   = np.gradient(self._ux, gs, axis=1)
            if self._dux_dy   is None: self._dux_dy   = np.gradient(self._ux, gs, axis=2)
            if self._dux_deta is None: self._dux_deta = np.gradient(self._ux, es, axis=3)
            if self._duy_dx   is None: self._duy_dx   = np.gradient(self._uy, gs, axis=1)
            if self._duy_dy   is None: self._duy_dy   = np.gradient(self._uy, gs, axis=2)
            if self._duy_deta is None: self._duy_deta = np.gradient(self._uy, es, axis=3)
            if self._duz_dx   is None: self._duz_dx   = np.gradient(self._uz, gs, axis=1)
            if self._duz_dy   is None: self._duz_dy   = np.gradient(self._uz, gs, axis=2)
            if self._duz_deta is None: self._duz_deta = np.gradient(self._uz, es, axis=3)

        # 3. Domain metadata
        self.t0       = float(np.amin(self._tspace))
        self.tf       = float(np.amax(self._tspace))
        self.timestep = float(self._tspace[-1] - self._tspace[-2])
        self.xmin     = float(np.amin(self._xspace))
        self.xmax     = float(np.amax(self._xspace))
        self.ymin     = self.xmin
        self.ymax     = self.xmax
        self.gridstep = gs
        if self.boost_invariant:
            self.etasmin = 0.0
            self.etasmax = 0.0
        else:
            self.etasmin = float(np.amin(self._etas_space))
            self.etasmax = float(np.amax(self._etas_space))

    ##################################
    # Private interpolation dispatch #
    ##################################

    def _eval(self, arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate *arr* at *pts* (..., 4).

        In boost-invariant mode: arr is (NT,NX,NY),  eta_s column is dropped.
        In 3+1D mode:            arr is (NT,NX,NY,NETA), all 4 coords used.
        """
        if self.boost_invariant:
            return _eval_3d(arr, self._tspace, self._xspace, pts)
        else:
            return _eval_4d(arr, self._tspace, self._xspace, self._etas_space, pts)

    ######################################################################
    # Field callables (public API — identical signatures for both modes) #
    ######################################################################

    def temp(self, pts) -> np.ndarray:
        """Temperature [GeV] at pts = (..., [tau, x, y, eta_s])."""
        return self._eval(self._T, pts)

    def x_vel(self, pts) -> np.ndarray:
        """
        Lab-frame x-velocity.
        2+1D: ux / cosh(eta_s)   (Bjorken boost-invariant decomposition)
        3+1D: interpolated directly from stored ux field
        """
        p = np.asarray(pts, dtype=float)
        if self.boost_invariant:
            ux = self._eval(self._ux, p)
            eta_s = p[..., 3] if p.shape[-1] == 4 else 0.0
            return ux / np.cosh(eta_s)
        else:
            return self._eval(self._ux, p)

    def y_vel(self, pts) -> np.ndarray:
        """
        Lab-frame y-velocity.
        2+1D: uy / cosh(eta_s)
        3+1D: interpolated directly from stored uy field
        """
        p = np.asarray(pts, dtype=float)
        if self.boost_invariant:
            uy = self._eval(self._uy, p)
            eta_s = p[..., 3] if p.shape[-1] == 4 else 0.0
            return uy / np.cosh(eta_s)
        else:
            return self._eval(self._uy, p)

    def z_vel(self, pts) -> np.ndarray:
        """
        Lab-frame z-velocity.
        2+1D: tanh(eta_s)   (exact Bjorken result)
        3+1D: interpolated from stored uz field
        """
        p = np.asarray(pts, dtype=float)
        if self.boost_invariant:
            eta_s = p[..., 3] if p.shape[-1] == 4 else 0.0
            return np.tanh(eta_s)
        else:
            return self._eval(self._uz, p)

    def vel(self, pts) -> np.ndarray:
        """Total velocity magnitude sqrt(vx²+vy²+vz²)."""
        return np.sqrt(self.x_vel(pts)**2 + self.y_vel(pts)**2 + self.z_vel(pts)**2)

    # Temperature gradients

    def temp_grad_x(self, pts) -> np.ndarray:
        return self._eval(self._dT_dx, pts)

    def temp_grad_y(self, pts) -> np.ndarray:
        return self._eval(self._dT_dy, pts)

    def temp_grad_z(self, pts) -> np.ndarray:
        """
        2+1D: zero (boost-invariant)
        3+1D: interpolated d(T)/d(eta_s)
        """
        if self.boost_invariant:
            p = np.asarray(pts, dtype=float)
            return np.zeros(p.shape[:-1])
        else:
            return self._eval(self._dT_deta, pts)

    # Flow velocity gradient tensor

    def grad_x_u_x(self, pts) -> np.ndarray:
        return self._eval(self._dux_dx, pts)

    def grad_x_u_y(self, pts) -> np.ndarray:
        return self._eval(self._duy_dx, pts)

    def grad_x_u_z(self, pts) -> np.ndarray:
        if self.boost_invariant:
            p = np.asarray(pts, dtype=float)
            return np.zeros(p.shape[:-1])
        else:
            return self._eval(self._duz_dx, pts)

    def grad_y_u_x(self, pts) -> np.ndarray:
        return self._eval(self._dux_dy, pts)

    def grad_y_u_y(self, pts) -> np.ndarray:
        return self._eval(self._duy_dy, pts)

    def grad_y_u_z(self, pts) -> np.ndarray:
        if self.boost_invariant:
            p = np.asarray(pts, dtype=float)
            return np.zeros(p.shape[:-1])
        else:
            return self._eval(self._duz_dy, pts)

    def grad_z_u_x(self, pts) -> np.ndarray:
        if self.boost_invariant:
            p = np.asarray(pts, dtype=float)
            return np.zeros(p.shape[:-1])
        else:
            return self._eval(self._dux_deta, pts)

    def grad_z_u_y(self, pts) -> np.ndarray:
        if self.boost_invariant:
            p = np.asarray(pts, dtype=float)
            return np.zeros(p.shape[:-1])
        else:
            return self._eval(self._duy_deta, pts)

    def grad_z_u_z(self, pts) -> np.ndarray:
        """
        2+1D: 1 / (tau * cosh(eta_s))  — kinematic Bjorken term
        3+1D: interpolated d(uz)/d(eta_s)
        """
        p = np.asarray(pts, dtype=float)
        if self.boost_invariant:
            tau   = p[..., 0]
            eta_s = p[..., 3] if p.shape[-1] == 4 else 0.0
            return 1.0 / (tau * np.cosh(eta_s))
        else:
            return self._eval(self._duz_deta, pts)

    ##################
    # Domain helpers #
    ##################

    def xspace(self, resolution: int = 100, fraction: float = 1.0) -> np.ndarray:
        """Uniform array over [fraction·xmin, fraction·xmax] with *resolution* points."""
        lo = fraction * self.xmin
        hi = fraction * self.xmax
        return np.arange(lo, hi, (hi - lo) / resolution)

    def tspace(self, resolution: int = 100) -> np.ndarray:
        """Uniform array over [t0, tf] with *resolution* points."""
        return np.arange(self.t0, self.tf, (self.tf - self.t0) / resolution)

    ##########################
    # Temperature statistics #
    ##########################

    def _grid_points(self, time: float, resolution: int) -> np.ndarray:
        """
        Return a grid of query points at fixed *time* and eta_s=0,
        shaped (resolution, resolution, 3) for 2+1D or (resolution, resolution, 4) for 3+1D.
        """
        x_sp = self.xspace(resolution=resolution)
        x_coords, y_coords = np.meshgrid(x_sp, x_sp, indexing='ij')
        t_coords = np.full_like(x_coords, time)

        if self.boost_invariant:
            return np.transpose(np.array([t_coords, x_coords, y_coords]), (2, 1, 0))
        else:
            etas_coords = np.zeros_like(x_coords)
            return np.transpose(np.array([t_coords, x_coords, y_coords, etas_coords]), (2, 1, 0))

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
    # Plotting  (2+1D slice at fixed eta_s; works for both modes)
    # -----------------------------------------------------------------------

    def plot(self, time=None, temp_resolution=100, vel_resolution=100,
             grad_resolution=100, temptype='contour', veltype='stream',
             gradtype='stream', plot_temp=True, plot_vel=True, plot_grad=False,
             numContours=15, zoom=1, eta_s=0):
        if time is None:
            time = self.t0

        tempMax = self.max_temp()
        transposeAxes = (2, 1, 0)

        # Temperature
        x_space = self.xspace(resolution=temp_resolution, fraction=zoom)
        if plot_temp:
            x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
            t_coords = np.full_like(x_coords, time)
            points = np.transpose(np.array([t_coords, x_coords, y_coords]), transposeAxes)
            temp_points = self.temp(points)
        else:
            temp_points = 0

        # Velocities
        x_space_vel = self.xspace(resolution=vel_resolution, fraction=zoom)
        if plot_vel:
            vel_x_coords, vel_y_coords = np.meshgrid(x_space_vel, x_space_vel, indexing='ij')
            vel_etas = np.full_like(vel_x_coords, eta_s)
            vel_t = np.full_like(vel_x_coords, time)
            vel_points = np.transpose(
                np.array([vel_t, vel_x_coords, vel_y_coords, vel_etas]), transposeAxes
            )
            x_vels = self.x_vel(vel_points)
            y_vels = self.y_vel(vel_points)
        else:
            x_vels = y_vels = 0

        # Gradients
        x_space_grad = self.xspace(resolution=grad_resolution, fraction=zoom)
        if plot_grad:
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
            grad_x = grad_y = 0
            grad_max = 1.0

        # Render temperature
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

        # Render velocities
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

        # Render gradients
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


    ###########################
    # Freeze-out hypersurface #
    ###########################
    def find_freezeout_surface(self, T_switch: float= config.soft_transport.hydro.T_SWITCH,
                               force_recompute: bool = False) -> "FreezeoutSurface":
        """
        Locate (and cache) the freeze-out hypersurface T(tau,x,y,eta_s) = T_switch.

        The result is cached on this plasma object, keyed by T_switch,
        so repeated calls (e.g. many sample_thermal_parton() draws) are cheap.
        The cache is a plain dict of picklable FreezeoutSurface objects, so the
        plasma object remains fully pickle-able for parallelization.

        Parameters
        ----------
        T_switch : float
            Switching / freeze-out temperature [GeV]. Typically config.soft_transport.hydro.T_SWITCH.
        force_recompute : bool
            If True, bypass and overwrite any cached surface.

        Returns
        -------
        FreezeoutSurface
        """
        key = round(float(T_switch), 6)
        if not force_recompute and key in self._freezeout_cache:
            return self._freezeout_cache[key]

        surface = self._compute_freezeout_surface(T_switch)
        self._freezeout_cache[key] = surface
        return surface

    def clear_freezeout_cache(self) -> None:
        """Discard all cached freeze-out hypersurfaces."""
        self._freezeout_cache.clear()

    def _compute_freezeout_surface(self, T_switch: float=config.soft_transport.hydro.T_SWITCH) -> "FreezeoutSurface":
        gs = self.gridstep

        if self.boost_invariant:
            # T, ux, uy are eta_s-independent (Bjorken symmetry), so the
            # crossing is found once on the native (x,y) grid. phi_fo uses
            # ux,uy directly: the shared 1/cosh(eta_s) factor on vx,vy cancels
            # in atan2, so phi is eta_s-independent too. y_fo is not computed
            # here: it equals eta_s exactly, and is handled analytically
            # (sampled flat) in sample_thermal_parton.
            tau_fo, i_lo, frac, valid = _find_temperature_crossing(
                self._T, self._tspace, T_switch
            )
            vx_fo = _interp_at_crossing(self._ux, i_lo, frac)
            vy_fo = _interp_at_crossing(self._uy, i_lo, frac)

            x_grid, y_grid = np.meshgrid(self._xspace, self._xspace, indexing='ij')
            phi_fo = np.arctan2(vy_fo, vx_fo)

            cell_volume = np.where(valid, tau_fo * gs * gs, 0.0)
            weight = cell_volume

            return FreezeoutSurface(
                boost_invariant=True,
                T_switch=float(T_switch),
                xpos=x_grid, ypos=y_grid, etas=None,
                tau_fo=tau_fo, vx_fo=vx_fo, vy_fo=vy_fo, vz_fo=None,
                phi_fo=phi_fo, y_fo=None,
                weight=weight, valid=valid,
            )

        else:
            tau_fo, i_lo, frac, valid = _find_temperature_crossing(
                self._T, self._tspace, T_switch
            )
            vx_fo = _interp_at_crossing(self._ux, i_lo, frac)
            vy_fo = _interp_at_crossing(self._uy, i_lo, frac)
            vz_fo = _interp_at_crossing(self._uz, i_lo, frac)

            x_grid, y_grid, eta_grid = np.meshgrid(
                self._xspace, self._xspace, self._etas_space, indexing='ij'
            )
            phi_fo = np.arctan2(vy_fo, vx_fo)
            y_fo = np.arctanh(np.clip(vz_fo, -0.999999, 0.999999))

            es = float(self._etas_space[-1] - self._etas_space[-2])
            cell_volume = np.where(valid, tau_fo * gs * gs * es, 0.0)
            weight = cell_volume

            return FreezeoutSurface(
                boost_invariant=False,
                T_switch=float(T_switch),
                xpos=x_grid, ypos=y_grid, etas=eta_grid,
                tau_fo=tau_fo, vx_fo=vx_fo, vy_fo=vy_fo, vz_fo=vz_fo,
                phi_fo=phi_fo, y_fo=y_fo,
                weight=weight, valid=valid,
            )

    ###########################
    # Thermal parton sampling #
    ###########################
    def sample_thermal_parton(self, T_switch: float = config.soft_transport.hydro.T_SWITCH,
                              pid: int = 21,
                              n_samples: int = 1,
                              y_range: Optional[tuple] = None,
                              rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample the 3-momentum (px, py, pz) [GeV] of one or more "thermal partons"
        emitted from the freeze-out hypersurface at T_switch.

        - Momentum magnitude is drawn from a massless Bose-Einstein distribution
          (pid == 21, gluon) or Fermi-Dirac distribution (|pid| in 1..6, light
          quark), both at temperature T_switch.
        - Direction (phi, y) is drawn from a distribution over the freeze-out
          hypersurface cells, weighted by estimated particle density
          (~T_switch^3, via rho()) times each cell's proper volume. This makes
          the sampled direction follow the medium's flow field: cells are more
          likely to be picked where more freeze-out volume points.
        - In 2+1D (boost-invariant) mode, phi comes from the transverse flow at
          the sampled cell, while y (no finite-rapidity info exists in boost-
          invariant hydro) is drawn uniformly from y_range, which is required
          in that case.
        - In 3+1D mode, both phi and y are read directly from the sampled cell's
          local flow (y = arctanh(vz)).

        The underlying freeze-out hypersurface is computed once and cached on
        this plasma object (see find_freezeout_surface), so repeated calls are
        cheap.

        Parameters
        ----------
        T_switch  : float            Switching/freeze-out temperature [GeV].
        pid       : int              21 for gluon (Bose-Einstein),
                                      ±1,±2,±3,±4,±5,±6 for light quarks (Fermi-Dirac).
        n_samples : int              Number of partons to sample.
        y_range : (float, float)
            Required in boost-invariant mode. In Bjorken flow, fluid rapidity
            equals eta_s exactly, and the proper freeze-out volume element
            tau_fo*dx*dy*deta_s does not depend on eta_s — so phi and y are
            statistically independent, and y can be sampled flat over
            y_range without needing to grid eta_s at all. Ignored in 3+1D
            mode, where y is read directly from the sampled surface cell.
        rng       : np.random.Generator, optional

        Returns
        -------
        p : ndarray, shape (3,) if n_samples == 1 else (n_samples, 3)
            [px, py, pz] in GeV.
        """
        if rng is None:
            rng = np.random.default_rng()

        if pid == 21:
            stats = 'be'
        elif abs(pid) in (1, 2, 3, 4, 5, 6):
            stats = 'fd'
        else:
            raise ValueError(f"sample_thermal_parton: unsupported pid={pid}; "
                             f"use 21 (gluon) or |pid| in 1..6 (light quark).")

        surface = self.find_freezeout_surface(T_switch)
        phi_all, y_all, w_all, valid_all = surface.flat()

        total_w = w_all[valid_all].sum() if valid_all.any() else 0.0
        if not valid_all.any() or total_w <= 0:
            raise RuntimeError(
                f"No valid freeze-out surface found for T_switch={T_switch}."
            )

        probs = np.where(valid_all, w_all, 0.0)
        probs = probs / probs.sum()

        cell_idx = rng.choice(len(probs), size=n_samples, p=probs)
        phi = phi_all[cell_idx]

        if surface.boost_invariant:
            if y_range is None:
                raise ValueError(
                    "sample_thermal_parton: boost-invariant mode requires "
                    "y_range=(ymin, ymax) (fluid rapidity == eta_s, sampled "
                    "flat since the freeze-out weight is eta_s-independent)."
                )
            y = rng.uniform(y_range[0], y_range[1], size=n_samples)
        else:
            y = y_all[cell_idx]

        if stats == 'be':
            p_mag = _sample_be_momentum(T_switch, n_samples, rng)
        else:
            p_mag = _sample_fd_momentum(T_switch, n_samples, rng)

        E = p_mag
        pz = E * np.tanh(y)
        pT = E / np.cosh(y)
        px = pT * np.cos(phi)
        py = pT * np.sin(phi)

        p = np.stack([px, py, pz], axis=-1)
        return p[0] if n_samples == 1 else p


##########################################
# Factory helpers (public API preserved) #
##########################################

def tabulated_plasma(t_space, x_space, temp_values, x_vel_values, y_vel_values,
                     name=None, return_grids=False,
                     boost_invariant=True,
                     etas_space=None, z_vel_values=None):
    """
    Build a plasma_event from pre-computed grid arrays.

    2+1D (default):
        Arrays shaped (NT, NX, NX).  boost_invariant=True (default).

    3+1D:
        Arrays shaped (NT, NX, NX, NETA).
        Pass boost_invariant=False, etas_space=<1D array>, z_vel_values=<array>.
    """
    logging.debug('WARNING: Gradients of temp and flow not verified')

    if boost_invariant:
        plasma_object = plasma(
            boost_invariant=True,
            name=name,
            _tspace=np.asarray(t_space),
            _xspace=np.asarray(x_space),
            _T=np.asarray(temp_values),
            _ux=np.asarray(x_vel_values),
            _uy=np.asarray(y_vel_values),
        )
    else:
        if etas_space is None or z_vel_values is None:
            raise ValueError(
                "tabulated_plasma: 3+1D mode requires etas_space and z_vel_values."
            )
        plasma_object = plasma(
            boost_invariant=False,
            name=name,
            _tspace=np.asarray(t_space),
            _xspace=np.asarray(x_space),
            _etas_space=np.asarray(etas_space),
            _T=np.asarray(temp_values),
            _ux=np.asarray(x_vel_values),
            _uy=np.asarray(y_vel_values),
            _uz=np.asarray(z_vel_values),
        )

    if return_grids:
        return plasma_object, temp_values, x_vel_values, y_vel_values
    return plasma_object

def functional_plasma_2_1D(temp_func=None, x_vel_func=None, y_vel_func=None, name=None,
                           resolution=10, xmax=15, time=None, return_grids=False, tau0=0.5):
    """Build a 2+1D plasma_event by evaluating callable functions on a grid."""
    if time is None:
        t_space = np.linspace(tau0, 2 * xmax, resolution)
    else:
        t_space = np.linspace(tau0, time, resolution)
    x_space = np.linspace(-xmax, xmax, resolution)

    t_coords, x_coords, y_coords = np.meshgrid(t_space, x_space, x_space, indexing='ij')

    grid_shape = t_coords.shape

    def _eval_on_grid(func, *coords):
        """Evaluate func and guarantee output is a full grid-shaped array."""
        result = func(*coords)
        return np.broadcast_to(np.asarray(result, dtype=float), grid_shape).copy()

    temp_values  = _eval_on_grid(temp_func,  t_coords, x_coords, y_coords)
    x_vel_values = _eval_on_grid(x_vel_func, t_coords, x_coords, y_coords)
    y_vel_values = _eval_on_grid(y_vel_func, t_coords, x_coords, y_coords)

    return tabulated_plasma(t_space, x_space, temp_values, x_vel_values, y_vel_values,
                            name=name, return_grids=return_grids)

def functional_plasma_3_1D(temp_func=None, x_vel_func=None, y_vel_func=None, z_vel_func=None,
                            name=None, resolution=10, xmax=15, etasmax=5.0, time=None,
                            return_grids=False, tau0=0.5):
    """
    Build a 3+1D plasma_event by evaluating callable functions on a (tau, x, y, eta_s) grid.

    All four callables must accept four broadcastable array arguments:
        func(tau, x, y, eta_s) -> array

    Parameters
    ----------
    temp_func    : callable(tau, x, y, eta_s) -> temperature [GeV]
    x_vel_func   : callable(tau, x, y, eta_s) -> lab-frame vx
    y_vel_func   : callable(tau, x, y, eta_s) -> lab-frame vy
    z_vel_func   : callable(tau, x, y, eta_s) -> lab-frame vz
    name         : optional label stored on the plasma_event
    resolution   : number of grid points per fm along each spatial axis
    xmax         : half-width of the transverse grid [fm]
    etasmax      : half-width of the eta_s grid (symmetric about 0)
    time         : final proper time [fm]; defaults to 2*xmax
    return_grids : if True, also return the raw evaluated arrays
    tau0         : initial proper time [fm]
    """
    if time is None:
        t_space = np.linspace(tau0, 2 * xmax, resolution)
    else:
        t_space = np.linspace(tau0, time, resolution)
    x_space    = np.linspace(-xmax,    xmax,   resolution)
    etas_space = np.linspace(-etasmax, etasmax, resolution)

    t_coords, x_coords, y_coords, etas_coords = np.meshgrid(
        t_space, x_space, x_space, etas_space, indexing='ij'
    )

    grid_shape = t_coords.shape

    def _eval_on_grid(func, *coords):
        """Evaluate func and guarantee output is a full grid-shaped array."""
        result = func(*coords)
        return np.broadcast_to(np.asarray(result, dtype=float), grid_shape).copy()

    temp_values  = _eval_on_grid(temp_func,  t_coords, x_coords, y_coords, etas_coords)
    x_vel_values = _eval_on_grid(x_vel_func, t_coords, x_coords, y_coords, etas_coords)
    y_vel_values = _eval_on_grid(y_vel_func, t_coords, x_coords, y_coords, etas_coords)
    z_vel_values = _eval_on_grid(z_vel_func, t_coords, x_coords, y_coords, etas_coords)

    return tabulated_plasma(
        t_space=t_space, x_space=x_space, etas_space=etas_space,
        temp_values=temp_values,
        x_vel_values=x_vel_values, y_vel_values=y_vel_values, z_vel_values=z_vel_values,
        name=name, return_grids=return_grids,
        boost_invariant=False
    )


@dataclass(slots=True)
class FreezeoutSurface:
    """
    boost_invariant=True:
        A single (NX,NY) sheet in x,y. tau_fo, vx_fo, vy_fo, phi_fo, weight,
        valid are shape (NX,NY). etas and y_fo are None: in Bjorken flow,
        y_fo == eta_s exactly, and the proper cell volume tau_fo*dx*dy*deta_s
        is independent of eta_s, so the (phi, y) distribution factorizes as
        [this surface's phi distribution] x [[a flat distribution in y]].
        No eta_s grid is needed — see sample_thermal_parton.

    boost_invariant=False:
        A (NX,NY,NETA) sheet. All arrays (incl. vz_fo, y_fo) are shape (NX,NY,NETA).
    """
    boost_invariant: bool
    T_switch: float

    xpos: np.ndarray
    ypos: np.ndarray
    etas: Optional[np.ndarray]

    tau_fo: np.ndarray
    vx_fo:  np.ndarray
    vy_fo:  np.ndarray
    vz_fo:  Optional[np.ndarray]

    phi_fo: np.ndarray
    y_fo:   Optional[np.ndarray]

    weight: np.ndarray
    valid:  np.ndarray

    def flat(self):
        phi = self.phi_fo.ravel()
        y = self.y_fo.ravel() if self.y_fo is not None else None
        w = self.weight.ravel()
        v = self.valid.ravel()
        return phi, y, w, v