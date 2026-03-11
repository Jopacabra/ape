import numpy as np
import logging
import matplotlib.pyplot as plt
import math

import config

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, Sequence, TypeVar, Generic

class StopEvolve(Exception):
    """ Raise to end parton evolution early. """

# Species table
# Masses in GeV. PDG IDs follow standard conventions.
# WARNING!!! The masses for diquarks are just the nominal mass in GeV from Pythia's data tables.
# WARNING!!! This is not the same as the masses used for the quarks and gauge bosons in APE evolution
# WARNING!!! Ape's mass is passed back to Pythia for hadronization... That's not right for the light quarks!
_PARTICLE_SPECIES: dict[int, dict[str, float | int | str]] = {
    # Light quarks
    1: {"name": "d", "m": 0.00467, "m0":0.330},
    -1: {"name": "dbar", "m": 0.00467, "m0":0.330},
    2: {"name": "u", "m": 0.00216, "m0":0.330},
    -2: {"name": "ubar", "m": 0.00216, "m0":0.330},
    3: {"name": "s", "m": 0.0934, "m0":0.500},
    -3: {"name": "sbar", "m": 0.0934, "m0":0.500},
    # Heavy quarks
    4: {"name": "c", "m": 1.50000, "m0": 1.50000},
    -4: {"name": "cbar", "m": 1.50000, "m0": 1.50000},
    5: {"name": "b", "m": 4.80000, "m0": 4.80000},
    -5: {"name": "bbar", "m": 4.80000, "m0": 4.80000},
    6: {"name": "t", "m": 173.00000, "m0": 173.00000},
    -6: {"name": "tbar", "m": 173.00000, "m0": 173.00000},
    # Gauge bosons
    21: {"name": "g", "m": 0.0, "m0": 0.0},
    22: {"name": "gamma", "m": 0.0, "m0": 0.0},
    23: {"name": "Z0", "m": 0.0, "m0": 0.0},
    24: {"name": "W+", "m": 0.0, "m0": 0.0},
    # Diquark states -- beam remnants -- Only have pythia constituent quark masses
    1103: {"name": "dd_1", "m": None, "m0": 0.77133},
    -1103: {"name": "dd_1bar", "m": None, "m0": 0.77133},
    2101: {"name": "ud_0", "m": None, "m0": 0.57933},
    -2101: {"name": "ud_0bar", "m": None, "m0": 0.57933},
    2103: {"name": "ud_1", "m": None, "m0": 0.77133},
    -2103: {"name": "ud_1bar", "m": None, "m0": 0.77133},
    2203: {"name": "uu_1", "m": None, "m0": 0.77133,},
    -2203: {"name": "uu_1bar", "m": None, "m0": 0.77133,},
    3101: {"name": "sd_0", "m": None, "m0": 0.80473},
    -3101: {"name": "sd_0bar", "m": None, "m0": 0.80473},
    3103: {"name": "sd_1", "m": None, "m0": 0.92953},
    -3103: {"name": "sd_1bar", "m": None, "m0": 0.92953},
    3201: {"name": "su_0", "m": None, "m0": 0.80473},
    -3201: {"name": "su_0bar", "m": None, "m0": 0.80473},
    3203: {"name": "su_1", "m": None, "m0": 0.92953},
    -3203: {"name": "su_1bar", "m": None, "m0": 0.92953},
    3303: {"name": "ss_1", "m": None, "m0": 1.09361},
    -3303: {"name": "ss_1bar", "m": None, "m0": 1.09361},
}

@dataclass(frozen=True, slots=True)
class ParticleDelta:
    """
    Momentum transfer object
    """
    dpx: float = 0.0
    dpy: float = 0.0
    dpz: float = 0.0

# Flexible dataclass defining a parton's properties.
@dataclass(slots=True)
class Particle:
    """
    Parton state in Milne coordinates (tau, x, y, eta_s) with 4-momentum (E, p_x, p_y, p_z).

    Conventions:
      - Natural units (c = 1)
      - Energies/momenta in GeV
      - Coordinates: tau (time-like), x/y (transverse), eta_s (space-time rapidity)

    Design:
      - Identity: `id` (PDG species id), `m` (mass)
      - Evolving state: tau, x, y, etas, p_x, p_y, p_z
      - Derived kinematics: pT, mT, E, rapidity
    """
    logging.debug("Creating new parton...")

    # constructor inputs (what you want to pass)
    id: int
    px: float
    py: float
    pz: float
    tau: float = None
    x: float = 0.0
    y: float = 0.0
    etas: float = 0.0
    scalein: float = None
    col: int = None
    acol: int = None
    tag: int = None

    # filled in post-init
    m: float | None = field(init=False)
    m0: float = field(init=False)  # Pythia m0 mass, e.g. constituent quark mass
    name: str = field(init=False)

    # per-step state history (each entry is (tau, x, y, etas, px, py, pz))
    history: list[tuple[float, float, float, float, float, float, float]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    # initial snapshot
    tau_0: float = field(init=False)
    x_0: float = field(init=False)
    y_0: float = field(init=False)
    etas_0: float = field(init=False)
    px_0: float = field(init=False)
    py_0: float = field(init=False)
    pz_0: float = field(init=False)

    def __post_init__(self) -> None:
        # validate parton species and set derived identity fields
        try:
            info = _PARTICLE_SPECIES[self.id]
        except KeyError as e:
            logging.exception(e)
            raise ValueError(f"Unknown pid={self.id}") from e
        self.m0 = float(info["m0"])
        self.name = str(info["name"])
        try:
            self.m = float(info["m"])
        except Exception as e:
            # logging.exception(e)
            logging.debug(e)
            logging.debug(f"Missing m for id={self.id}, using m0")
            self.m = None
        
        # set default tau to tau_fs, if none provided
        if self.tau is None:
            self.tau = config.transport.hydro.TAU_FS

        # set initial snapshot
        self.tau_0 = float(self.tau)
        self.x_0 = float(self.x)
        self.y_0 = float(self.y)
        self.etas_0 = float(self.etas)
        self.px_0 = float(self.px)
        self.py_0 = float(self.py)
        self.pz_0 = float(self.pz)

        # initialize history with the initial state
        self._log_state()

        # Log particle creation
        logging.debug(
            "Initialized Parton(id=%s, name=%s) x,y=(%.6g, %.6g) p=(%.6g, %.6g, %.6g) tau=%.6g etas=%.6g",
            self.id, self.name, self.x, self.y, self.px, self.py, self.pz, self.tau, self.etas,
        )

    ################
    # State memory #
    ################
    def _log_state(self) -> None:
        """Append current evolving state to `history`."""
        self.history.append(
            (
                float(self.tau),
                float(self.x),
                float(self.y),
                float(self.etas),
                float(self.px),
                float(self.py),
                float(self.pz),
            )
        )

    ##############
    # Kinematics #
    ##############
    """
    Derived kinematics (computed on-demand so they're never out of sync)
    """

    @property
    def pT(self) -> float:
        return float(np.hypot(self.px, self.py))

    @property
    def mT(self) -> float:
        try:
            return float(np.hypot(self.m, self.pT))
        except Exception as e:
            # logging.exception(e)
            # logging.debug(f"Missing m for id={self.id}, computing mT w/ m0.")
            return self.mT0


    @property
    def mT0(self) -> float:
        return float(np.hypot(self.m0, self.pT))

    @property
    def E(self) -> float:
        # on-shell energy
        try:
            return float(np.sqrt(self.pT * self.pT + self.pz * self.pz + self.m * self.m))
        except Exception as e:
            # logging.exception(e)
            # logging.debug(f"Missing m for id={self.id}, computing rapidity w/ m0.")
            return self.E0

    @property
    def E0(self) -> float:
        # on-shell energy using Pythia constituent mass
        return float(np.sqrt(self.pT * self.pT + self.pz * self.pz + self.m0 * self.m0))

    @property
    def rap(self) -> float:
        # momentum rapidity y = 1/2 ln((E+pz)/(E-pz))
        E = self.E
        denom = E - self.pz
        if denom <= 0.0:
            # protect against numerical issues at extreme boosts
            denom = np.finfo(float).tiny
        return float(0.5 * np.log((E + self.pz) / denom))

    def pathlength_since(self, tau_0) -> float:
        """
        Get the 3D pathlength traveled since the given longitudinal proper time in fm
        """

        # Get trajectory history
        traj = np.asarray(self.history, dtype=float)
        tau = traj[:, 0]
        x = traj[:, 1][tau >= tau_0]  # Indexed for all steps since tau_0.
        y = traj[:, 2][tau >= tau_0]
        z = (tau * np.sinh(traj[:, 3]))[tau >= tau_0]

        # Get pathlength length in each coordinate
        x_length = np.sum(np.abs(x[1:] - x[:-1]))  # Sum over all step lengths (absolute value!!!)
        y_length = np.sum(np.abs(y[1:] - y[:-1]))
        z_length = np.sum(np.abs(z[1:] - z[:-1]))

        # Return total pathlength
        return np.sqrt(x_length**2 + y_length**2 + z_length**2)  # [fm], same units as history

    def next_pathlength(self, dtau, cart=False):
        """
        Get the pathlength traveled in the next timestep
        """
        # Compute relativistic velocities
        mT = self.mT
        rap_diff = self.etas - self.rap
        betatau = float(1.0)  # Unitless
        betax = float(self.px / (np.cosh(rap_diff) * mT))  # Unitless
        betay = float(self.py / (np.cosh(rap_diff) * mT))  # Unitless
        betaetas = float(-np.tanh(rap_diff) / self.tau)  # Units: fm^-1

        delta_tau = betatau * dtau
        delta_x = betax * dtau
        delta_y = betay * dtau
        delta_etas = betaetas * dtau

        if cart:  # Return cartesian coordinate steps
            delta_t = self.tau * np.cosh(delta_etas)  # Assumes small change in tau
            delta_z = self.tau * np.sinh(delta_etas)  # Assumes small change in tau
            return delta_t, delta_x, delta_y, delta_z
        else:  # Return Milne coordinate steps
            return delta_tau, delta_x, delta_y, delta_etas


    ###########################
    # Freestreaming evolution #
    ###########################

    def prop(self, dtau: float = 0.0) -> None:
        """
        Free-streaming update in Milne coordinates.
        """
        dtau = float(dtau)
        if dtau == 0.0:
            return

        if self.tau <= 0.0:
            raise ValueError(f"prop() requires tau > 0, got tau={self.tau}")

        # Compute relativistic steps
        delta_tau, delta_x, delta_y, delta_etas = self.next_pathlength(dtau, cart=False)

        # Update spacetime state variables
        self.tau = float(self.tau + delta_tau)  # Units: [fm]
        self.x = float(self.x + delta_x)  # Units: [fm]
        self.y = float(self.y + delta_y)  # Units: [fm]
        self.etas = float(self.etas + delta_etas)  # Unitless: [fm^-1] * [fm] // unitless hyperbolic angle thing

        # if np.abs(self.etas) > config.jet.RAP_MAX_EVOLVE:
        #     # self.etas = np.sign(self.etas) * config.jet.RAP_MAX_EVOLVE
        #     # self._log_state()
        #     raise StopEvolve(f"|etas|={np.abs(self.etas)} > RAP_MAX_EVOLVE={config.jet.RAP_MAX_EVOLVE}")
        # elif np.abs(self.rap) > config.jet.RAP_MAX_EVOLVE:
        #     # self._log_state()
        #     raise StopEvolve(f"|rap|={np.abs(self.rap)} > RAP_MAX_EVOLVE={config.jet.RAP_MAX_EVOLVE}")

        # Log state to history
        self._log_state()

    #############
    # Utilities #
    #############

    def copy(self) -> "Particle":
        return Particle(**self.to_kwargs())

    def spawn_child(self, **overrides: Any) -> "Particle":
        data = self.to_kwargs()
        data.update(overrides)
        return Particle(**data)

    def to_kwargs(self) -> dict[str, Any]:
        """Round-trippable constructor kwargs (matches Parton(...) signature)."""
        return {
            "id": self.id,
            "px": self.px,
            "py": self.py,
            "pz": self.pz,
            "tau": self.tau,
            "x": self.x,
            "y": self.y,
            "etas": self.etas,
            "tag": self.tag,
        }

    def to_dict(self) -> dict[str, Any]:
        # stable serialization for records / analysis
        return {
            "tau": self.tau,
            "x": self.x,
            "y": self.y,
            "etas": self.etas,
            "px": self.px,
            "py": self.py,
            "pz": self.pz,
            "id": self.id,
            "m": self.m,
            "tag": self.tag,
            "tau_0": self.tau_0,
            "x_0": self.x_0,
            "y_0": self.y_0,
            "etas_0": self.etas_0,
            "px_0": self.px_0,
            "py_0": self.py_0,
            "pz_0": self.pz_0,
            "scalein": self.scalein,
            "col": self.col,
            "acol": self.acol,
        }

    ###########################
    # Useful property queries #
    ###########################

    @property
    def p3(self):
        return np.array([self.px, self.py, self.pz])

    @property
    def coords(self):
        """Conveniently returns the current coordinates for calling a plasma.plasma_event object's properties"""
        return np.array([self.tau, self.x, self.y, self.etas])

    @property
    def isq(self):
        """Returns true if the parton is a quark"""
        if np.abs(self.id) < 7:
            return True
        else:
            return False

    @property
    def isg(self):
        """Returns true if the parton is a quark"""
        if self.id == 21:
            return True
        else:
            return False

    @property
    def isEWB(self):
        """Returns true if the parton is an electro-weak boson"""
        if self.id in [22, 23, 24]:
            return True
        else:
            return False

    ###########################
    # Kinematic Manipulations #
    ###########################

    def azimuthal_rotate(self, phi, initial=False):
        """Rotate all particles' momenta by phi radians in the transverse plane."""

        # Current
        old_px = self.px
        old_py = self.py
        self.px = float(old_px * np.cos(phi) - old_py * np.sin(phi))
        self.py = float(old_px * np.sin(phi) + old_py * np.cos(phi))

        # Initial
        if initial:
            self.px_0 = float(self.px)
            self.py_0 = float(self.py)

        # Future: Rotate coordinates in record

    def dp(self, dpx: float =0, dpy: float =0, dpz: float =0):
        """Add some 3-momentum to the parton"""
        self.px = float(self.px + dpx)
        self.py = float(self.py + dpy)
        self.pz = float(self.pz + dpz)

    def apply_deltap(self, delta: ParticleDelta):
        """Add some 3-momentum to the parton using a ParticleDelta object"""
        self.px = float(self.px + delta.dpx)
        self.py = float(self.py + delta.dpy)
        self.pz = float(self.pz + delta.dpz)

    def thermal_sample(self):
        """
        Sample a Fermi-Dirac (fermions) or Bose-Einstein (bosons) distribution at the freezeout temperature for some
        3 momentum, given pid of parton. Uses rejection sample.
        """
        # Get random points
        rng = np.random.default_rng()
        for j in range(100):
            num_samples = 1000
            E_samps = rng.uniform(0, 1, num_samples)  # Random energies, maximum 1 GeV.
            P_samps = rng.uniform(0, 1, num_samples)

            # Compute distribution values for energy values
            if self.isq:  # Use fermion dist. -- Fermi-Dirac distribution
                dist = (1 / (np.exp(-E_samps / config.transport.hydro.T_SWITCH) + 1))
            elif self.isg or self.isEWB:  # Use boson dist. -- Bose-Einstein distribution
                dist = (1 / (np.exp(-E_samps / config.transport.hydro.T_SWITCH) - 1))
            else:
                raise ValueError(f"Unknown parton statistics {self.id}")
            dist = dist / np.amax(dist)  # Normalize largest value to 1.

            i = 0
            for i in range(num_samples):
                if P_samps[i] <= dist[i]:  # Accept below or at curve
                    energy = E_samps[i]
                    break
                i += 1

            # Get num_points random unit 3-vectors for the direction
            rng = np.random.default_rng()
            new_p = rng.uniform(-1, 1, 3)  # Sample a random direction
            new_p = new_p / np.linalg.norm(new_p)  # Normalize
            new_p = np.sqrt(energy**2 - (self.m**2)) * new_p  # Scale momenta using appropriate mass on-shell condition

            # Set momentum to new momentum.
            self.px = new_p[0]
            self.py = new_p[1]
            self.pz = new_p[2]

            return True

        raise Exception("Failed to sample particle momentum.")

    ###########################
    # Convenient constructors #
    ###########################
    @classmethod
    def from_pythia(
            cls,
            p,
            *,
            tau: Optional[float] = None,
            x: float = 0.0,
            y: float = 0.0,
            etas: float = 0.0,
            tag: Optional[int] = None,
    ) -> "Particle":
        """
        Build a `Particle` from a single Pythia particle handle/object `p`.

        Usage:
            part = Particle.from_pythia(pythia_particle, tau=..., etas=...)
        """
        return cls(
            id=int(p.id()),
            px=float(p.px()),
            py=float(p.py()),
            pz=float(p.pz()),
            tau=tau,  # if None, Particle will choose its default in __post_init__
            x=float(x),
            y=float(y),
            etas=float(etas),
            scalein=float(p.scale()),
            col=int(p.col()),
            acol=int(p.acol()),
            tag=tag,
        )

ParticleT = TypeVar("ParticleT")
@dataclass(slots=True)
class EventRecord(Generic[ParticleT]):
    """ Container for a single event's particles + event-level metadata.
    Intended use:
      - store a small collection of Particle objects
      - carry event metadata (weight, ids, generator settings, etc.)
      - provide a stable interface between generator/evolver/hadronizer steps
    """

    particles: list[ParticleT]
    weight: Optional[float] = None
    event_id: Optional[int] = None
    event_seed: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # defensive: ensure `particles` is a real list (not a generator)
        if not isinstance(self.particles, list):
            self.particles = list(self.particles)

        n = len(self.particles)

        if self.weight is not None:
            self.weight = float(self.weight)

        if self.event_id is not None:
            self.event_id = int(self.event_id)

    # Collection-like convenience
    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self) -> Iterator[ParticleT]:
        return iter(self.particles)

    def __getitem__(self, idx: int) -> ParticleT:
        return self.particles[idx]

    # Mutators / utilities
    def append(self, p: ParticleT) -> None:
        self.particles.append(p)

    def extend(self, ps: Iterable[ParticleT]) -> None:
        ps_list = list(ps)
        self.particles.extend(ps_list)

    def copy(self, *, deep_particles: bool = True) -> "EventRecord[ParticleT]":
        """
        Copy the event record.

        deep_particles:
          - False: keep same particle objects
          - True: call `.copy()` on each particle
        """
        if deep_particles:
            new_particles = [p.copy() for p in self.particles]  # type: ignore[attr-defined]
        else:
            new_particles = list(self.particles)

        return EventRecord(
            particles=new_particles,
            weight=self.weight,
            event_id=self.event_id,
            meta=dict(self.meta),
        )

    def azimuthal_rotate(self, phi, initial=False):
        """Rotate all particles by phi radians in the transverse plane."""
        for i in range(len(self.particles)):
            self.particles[i].azimuthal_rotate(phi, initial=initial)

    def set_prod_point(self, x, y):
        """Rotate all particles by phi radians in the transverse plane."""
        for i in range(len(self.particles)):
            self.particles[i].x_0 = x
            self.particles[i].y_0 = y

    @classmethod
    def from_particles(
            cls,
            particles: Sequence[ParticleT],
            *,
            weight: Optional[float] = None,
            event_id: Optional[int] = None,
            meta: Optional[dict[str, Any]] = None,
    ) -> "EventRecord[ParticleT]":
        return cls(
            particles=list(particles),
            weight=weight,
            event_id=event_id,
            meta={} if meta is None else dict(meta),
        )