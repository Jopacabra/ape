from dataclasses import dataclass, field
import copy
from typing import Any, Iterable, Iterator, Optional, Sequence, TypeVar, Generic
import math
import logging

import numpy as np

import config
import utilities

class StopEvolve(Exception):
    """ Raise to end parton evolution early. """

# Species table
# Masses in GeV. PDG IDs follow standard conventions.
# WARNING!!! The masses for diquarks are just the constituent mass in GeV from Pythia's data tables.
# WARNING!!! This is not the same as the masses used for the quarks and gauge bosons in APE evolution
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
    # Pythia internal
    90: {"name": "event", "m": None, "m0": 0.0},
    # Beam hadrons
    2212: {"name": "p+", "m": None, "m0": 0.93827},
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
    tag: int | None = None

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
    thermalized: bool = False

    # Shower history information
    index : int = None # unique identifier for this parton in the shower -- usually from Pythia index
    status : int = 23
    mother1 : int = 0
    mother2: int = 0
    daughter1 : int = 0
    daughter2: int = 0

    # Fragmentation information
    fragz : float = None
    fragz0 : float = None

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
            logging.debug(f"Missing m for id={self.id}, using m0.")
            # logging.debug(e)
            self.m = None
        
        # set default tau to tau_fs, if none provided
        if self.tau is None:
            self.tau = config.soft_transport.all.TAU_FS

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
            "Initialized Particle (status=%s) (id=%s, name=%s) x,y=(%.6g, %.6g) p=(%.6g, %.6g, %.6g) tau=%.6g etas=%.6g",
            self.status, self.id, self.name, self.x, self.y, self.px, self.py, self.pz, self.tau, self.etas,
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
        pz = self.pz
        numer = E + pz
        denom = E - pz

        finfo = np.finfo(float)

        with np.errstate(over='ignore'):  # Intentionally getting overflow values
            # --- lower bound: division would overflow (y -> +inf) ---
            lower_threshold = np.abs(numer) / finfo.max
            overflow_case = np.abs(denom) <= np.maximum(lower_threshold, finfo.tiny)

            # --- upper bound: division would underflow to 0 (y -> -inf) ---
            upper_threshold = np.abs(numer) * finfo.max  # denom >> numer
            underflow_case = np.abs(denom) >= np.minimum(upper_threshold, finfo.max)

        if numer == 0.0:
            return np.nan  # This case covers the "event as a whole" particle.
        if overflow_case:
            return math.copysign(np.inf, pz)  # Beam particles
        elif underflow_case:
            return np.nan  # This case is not physically meaningful
        else:
            return 0.5 * np.log(numer / denom)

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
        return copy.deepcopy(self)

    def set_decayed(self, daughter1=None, daughter2=None):
        """ Set status to negative """
        self.status = int((-1) * abs(self.status))
        if daughter1 is not None:
            self.daughter1 = daughter1
        if daughter2 is not None:
            self.daughter2 = daughter2

    def spawn_child(self, **overrides: Any) -> "Particle":
        child = copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(child, k, v)
        return child

    def printout(self) -> dict[str, Any]:
        """Printable state output for debugging purposes."""
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
        """
        Stable serialization for records / analysis.

        Note that any field included here will be automatically included in the output dataframes, if written.
        """
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
            "fragz": self.fragz,
            "fragz0": self.fragz0,
        }

    ###########################
    # Useful property queries #
    ###########################

    @property
    def p3(self):
        return np.array([self.px, self.py, self.pz])

    @property
    def p30(self):
        return np.array([self.px_0, self.py_0, self.pz_0])

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
        for j in range(100):
            num_samples = 1000
            E_samps = utilities.rng.uniform(0, 1, num_samples)  # Random energies, maximum 1 GeV.
            P_samps = utilities.rng.uniform(0, 1, num_samples)

            # Compute distribution values for energy values
            if self.isq:  # Use fermion dist. -- Fermi-Dirac distribution
                dist = (1 / (np.exp(E_samps / config.soft_transport.hydro.T_SWITCH) + 1))
            elif self.isg or self.isEWB:  # Use boson dist. -- Bose-Einstein distribution
                dist = (1 / (np.exp(E_samps / config.soft_transport.hydro.T_SWITCH) - 1))
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
            new_p = utilities.rng.uniform(-1, 1, 3)  # Sample a random direction
            new_p = new_p / np.linalg.norm(new_p)  # Normalize
            new_p = energy * new_p  # Scale momenta -- this is exclusively the kinetic energy associated w/ 3-momenta

            # Set momentum to new momentum.
            self.px = new_p[0]
            self.py = new_p[1]
            self.pz = new_p[2]

            # Mark particle thermalized
            self.thermalized = True

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
            tag: Optional[int] = None
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
            status=int(p.status()),
            mother1=int(p.mother1()),
            mother2=int(p.mother2()),
            daughter1=int(p.daughter1()),
            daughter2=int(p.daughter2())
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
    event_tau0: float = 0.0
    event_x0: float = 0.0
    event_y0: float = 0.0
    event_etas0: float = 0.0
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

    def spawn_child_particles(self):
        original_length = len(self.particles)
        logging.debug(f"Spawning child particles for event {self.event_id}")
        logging.debug(f"Original event record: {original_length} particles")
        for p in self.particles[0:]:
            # If the particle is still alive, spawn a child particle
            if p.status > 0:
                # Get current length of event record
                n = len(self.particles)

                # Update daughter tags and status of the original particle to include the child
                p.status = int(-1*p.status)
                p.daughter1 = n
                p.daughter2 = n

                # Create the new child particle as a copy, updating mothers, daughters, and tag
                child = p.spawn_child(mother1=p.tag, mother2=p.tag, daughter1=0, daughter2=0, tag=n, status=23)

                # Append child particle to event record
                self.append(child)

        logging.debug(f"Final event record: {len(self.particles)} particles")

    def spawn_radiation(self, parent_tag, color_target_tag, k, coords) -> "Particle":
        """
        Spawn a radiated gluon at the given spacetime position with lab frame momentum k
        Adjust children of parent specified with "parent_tag". Add the radiated gluon and color rotate
        the particle specified with "color_target_tag" appropriately.
        """
        color_target = self.particles[color_target_tag]

        # Radiation is always a live gluon at this spacetime position at the end of the record.
        rad_tag = len(self.particles)

        # Color target is now a simple decay product (https://pythia.org//latest-manual/ParticleProperties.html)
        color_target.mother2 = 0

        # Radiation has color relations with color_target -- perform color rotation appropriately
        if color_target.isq:
            if color_target.id > 0:  # Particle is a quark, it carries a color
                emission_col = color_target.col
                new_color = color_target.col + 1  # only one color available, so we can just increment
                emission_acol = new_color
                color_target.col = emission_acol
            else:  # Particle is an antiquark, it carries an anticolor
                new_color = color_target.acol + 1  # only one color available, so we can just increment
                emission_col = new_color
                emission_acol = color_target.acol
                color_target.acol = emission_col
        elif color_target.isg:
            a = utilities.rng.choice([0,1])
            # Two colors available -- we increment the higher one to avoid possible singlet
            new_color = np.amax([color_target.col, color_target.acol]) + 1
            if a == 0:
                emission_col = color_target.col
                emission_acol = new_color
                color_target.col = emission_acol
            elif a == 1:
                emission_col = new_color
                emission_acol = color_target.acol
                color_target.acol = emission_col
            else:
                print("Uh-oh...")
        else:
            emission_col = 0
            emission_acol = 0

        emission = Particle(
            id=21,  # Gluon
            px=k[0],  # Has k momentum
            py=k[1],
            pz=k[2],
            tau=coords[0],  # Current spacetime position
            x=coords[1],
            y=coords[2],
            etas=coords[3],
            scalein=np.sqrt(np.dot(k,k)),  # Scale of emission ??? Need to check this.
            col=emission_col,  # Splits color from parent
            acol=emission_acol,
            tag=rad_tag,  # New tag, since it is unplaced in the event record
            status=23,  # standard Pythia status for hadronization
            mother1=parent_tag,  # Radiated by this particle
            mother2=0,
            daughter1=0,  # No radiation from this particle yet, so no daughters
            daughter2=0,
        )

        # Add radiation to event record and update parent particle's daughters
        self.append(emission)
        if self.particles[parent_tag].daughter1 == 0:  # This is the first daughter particle
            self.particles[parent_tag].daughter1 = rad_tag
        else:  # This is a subsequent daughter. daughter2 is the final index of radiated particles, so iteratively set
            self.particles[parent_tag].daughter2 = rad_tag

        return emission

    def proximity_color(
            self,
            parton_indices: Optional[Sequence[int]] = None,
            remnant_id: int = 1,
            remnant_pz: float = 50.0,
            remnant_pt: float = 0.2,
    ) -> None:
        """
        Perform proximity-based repairing of parton color/anticolor indices.

        This reimplements the core color-reconstruction logic used by JETSCAPE's
        ColorlessHadronization module (its PYTHIA Lund-string hadronization
        interface):

          1. Quarks/antiquarks are paired into "strings" by iteratively matching
             each unpaired quark to its angularly closest (min delta_R) unpaired
             partner.
          2. If no quarks are available (or a quark is left without a partner),
             fake beam-remnant quarks flying down +/-pz are attached to close
             off the string(s), mirroring JETSCAPE's remnant-momentum trick.
          3. Gluons are assigned to whichever string's two endpoints they are
             (on average) closest to.
          4. Each string's gluons are then chained together in nearest-neighbor
             order starting from one string endpoint, assigning sequential
             color/anticolor indices along the chain.
          5. Quark PDG ids are flipped, if needed, so that a nonzero `col`
             corresponds to a particle (positive id) and a nonzero `acol`
             corresponds to an antiparticle (negative id).

        delta_R here is computed from (rapidity, azimuthal angle) using each
        particle's momentum, analogous to fastjet's PseudoJet::delta_R() used
        in the original JETSCAPE implementation.

        Parameters
        ----------
        parton_indices:
            Indices into `self.particles` identifying which partons should have
            their colors repaired. Defaults to all currently-live (status > 0)
            quarks/antiquarks/gluons in the record.
        remnant_id:
            PDG id (must be a light quark, 1-6) used for any fake remnant quarks
            that need to be attached to close off unpaired strings.
        remnant_pz:
            Magnitude of the longitudinal momentum given to remnant quarks
            (they are shot down +pz or -pz).
        remnant_pt:
            Magnitude of the (px, py) components given to remnant quarks.
        """
        if parton_indices is None:
            parton_indices = [
                i for i, p in enumerate(self.particles)
                if p.status > 0 and (abs(p.id) <= 6 or p.id == 21)
            ]
        else:
            parton_indices = list(parton_indices)

        if not parton_indices:
            return

        pIn: list[int] = list(parton_indices)

        def phi_of(idx: int) -> float:
            p = self.particles[idx]
            return math.atan2(p.py, p.px)

        def delta_R(idx1: int, idx2: int) -> float:
            p1 = self.particles[idx1]
            p2 = self.particles[idx2]
            deta = p1.rap - p2.rap
            dphi = phi_of(idx1) - phi_of(idx2)
            dphi = (dphi + math.pi) % (2.0 * math.pi) - math.pi
            return math.hypot(deta, dphi)

        def add_remnant(pz_sign: float) -> int:
            """Attach a fake beam-remnant quark flying down +/-pz."""
            logging.warning("!" * 100)
            logging.warning("BEAM REMNANT REQUIRED!!!")
            logging.warning("!" * 100)
            new_particle = Particle(
                id=remnant_id,
                px=remnant_pt,
                py=remnant_pt,
                pz=pz_sign * remnant_pz,
                tau=self.event_tau0,
                x=self.event_x0,
                y=self.event_y0,
                etas=self.event_etas0,
                status=1,
                tag=len(self.particles),
            )
            new_idx = len(self.particles)
            self.append(new_particle)
            return new_idx

        # Identify quarks/antiquarks among the selected partons
        isquark: list[int] = [i for i in pIn if abs(self.particles[i].id) <= 6]
        nquarks = len(isquark)

        isdone: dict[int, bool] = {i: False for i in pIn}
        one_end: list[int] = []
        two_end: list[int] = []

        # If no quarks are present, seed a single string with two remnants
        if nquarks == 0:
            idx1 = add_remnant(+1.0)
            pIn.append(idx1)
            isquark.append(idx1)
            isdone[idx1] = True
            one_end.append(idx1)

            idx2 = add_remnant(-1.0)
            pIn.append(idx2)
            isquark.append(idx2)
            isdone[idx2] = True
            two_end.append(idx2)

        # Pair up quarks into strings, always matching to the closest
        # remaining unpaired quark (order matters, as in the original algo)
        for iq in range(len(isquark)):
            q_idx = isquark[iq]
            if isdone.get(q_idx, False):
                continue
            isdone[q_idx] = True
            one_end.append(q_idx)

            min_delR = float("inf")
            partner = None
            for jq in isquark:
                if jq == q_idx or isdone.get(jq, False):
                    continue
                dR = delta_R(q_idx, jq)
                if dR < min_delR:
                    min_delR = dR
                    partner = jq

            if partner is not None:
                isdone[partner] = True
                two_end.append(partner)
            else:
                # No partner available -- close off the string with a remnant
                new_idx = add_remnant(+1.0)
                pIn.append(new_idx)
                isquark.append(new_idx)
                isdone[new_idx] = True
                two_end.append(new_idx)

        nstrings = len(one_end)

        # Assign each gluon to whichever string it is (on average) closest to
        gluon_indices = [i for i in pIn if self.particles[i].id == 21]
        my_string: dict[int, int] = {}
        for g_idx in gluon_indices:
            min_delR = float("inf")
            best_string = 0
            for ns in range(nstrings):
                dR = 0.5 * (
                        delta_R(g_idx, one_end[ns]) + delta_R(g_idx, two_end[ns])
                )
                if dR < min_delR:
                    min_delR = dR
                    best_string = ns
            my_string[g_idx] = best_string

        # Build color chains along each string, linking gluons in
        # nearest-neighbor order and propagating alternating color/anticolor
        col: dict[int, int] = {}
        acol: dict[int, int] = {}
        lab_col = max(
            [1] + [self.particles[i].col or 0 for i in range(len(self.particles))]
            + [self.particles[i].acol or 0 for i in range(len(self.particles))]
        ) + 1

        gluon_done = {g: False for g in gluon_indices}

        for ns in range(nstrings):
            tquark = one_end[ns]
            if self.particles[tquark].id > 0:
                col[tquark] = lab_col
            else:
                acol[tquark] = lab_col
            lab_col += 1

            link = tquark
            while True:
                min_delR = float("inf")
                next_link = None
                for g_idx in gluon_indices:
                    if gluon_done[g_idx] or my_string[g_idx] != ns:
                        continue
                    dR = delta_R(link, g_idx)
                    if dR < min_delR:
                        min_delR = dR
                        next_link = g_idx

                if next_link is None:
                    break

                gluon_done[next_link] = True
                if col.get(link) == lab_col - 1:
                    col[next_link] = lab_col
                    acol[next_link] = lab_col - 1
                else:
                    col[next_link] = lab_col - 1
                    acol[next_link] = lab_col
                lab_col += 1
                link = next_link

            # Close off the string at its second quark end
            tail = two_end[ns]
            if col.get(link) == lab_col - 1:
                col[tail] = 0
                acol[tail] = lab_col - 1
            else:
                col[tail] = lab_col - 1
                acol[tail] = 0

        # Apply the computed color/anticolor indices
        for idx in pIn:
            self.particles[idx].col = col.get(idx, 0)
            self.particles[idx].acol = acol.get(idx, 0)

        # Fix quark identities to stay consistent with assigned color charge:
        # a nonzero `col` must belong to a particle, a nonzero `acol` to an
        # antiparticle.
        for q_idx in isquark:
            p = self.particles[q_idx]
            if col.get(q_idx, 0) != 0:
                if p.id < 0:
                    p.id = -p.id
            elif p.id > 0:
                p.id = -p.id
            else:
                continue

            # Refresh derived identity fields after any pid flip
            info = _PARTICLE_SPECIES[p.id]
            p.name = str(info["name"])
            p.m0 = float(info["m0"])
            try:
                p.m = float(info["m"])
            except Exception:
                p.m = None

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
        """Rotate all particles by phi radians in the transverse plane -- only for before evolution..."""
        for i in range(len(self.particles)):
            self.particles[i].azimuthal_rotate(phi, initial=initial)

    def set_prod_point(self, tau, x, y, etas):
        """
        Shift position of the hard scattering in the transverse plane -- only for before evolution...
        No history modification implemented.
        """

        # Iterate over particles
        for i in range(len(self.particles)):
            # Shift initial positions
            self.particles[i].tau_0 = self.particles[i].tau_0 - self.event_tau0 + tau
            self.particles[i].x_0 = self.particles[i].x_0 - self.event_x0 + x
            self.particles[i].y_0 = self.particles[i].y_0 - self.event_y0 + y
            self.particles[i].etas_0 = self.particles[i].etas_0 - self.event_etas0 + etas

            # Shift histories
            # ...

        # Shift event transverse coordinates
        self.event_tau0 = tau
        self.event_x0 = x
        self.event_y0 = y
        self.event_etas0 = etas


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