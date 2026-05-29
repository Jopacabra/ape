import lhapdf
import io
from contextlib import redirect_stdout
import logging

import numpy as np

import utilities

# Object that stores fragmentation function sets and exposes some sampling operations
class Fragger:
    """
    Fragmentation function set object, with logic to sample a fragmentation momentum fraction for a given parton.
    """
    def __init__(self, ff_name="JAM20-SIDIS_FF_hadron_nlo", seed=None):
        # constructor inputs
        self.ff_name = ff_name
        self.seed = seed

        # Load fragmentation function, capturing log text
        f = io.StringIO()
        with redirect_stdout(f):
            self.ff_set = lhapdf.mkPDFs(self.ff_name)
        logging.debug(f.getvalue())

        # Start rng
        self.rng = utilities.rng

    def pz(self, pid, z, pT):
        if pT < 1.14:
            pT = 1.14

        # We use the zeroth member -- this is the central value of the FF fit.
        return self.ff_set[0].xfxQ2(pid, z, pT**2)

    def frag(self, parton, i=False, num=1):
        # Get jet properties
        if i:
            parton_pT = float(np.hypot(parton.px_0, parton.py_0))
        else:
            parton_pT = parton.pT
        parton_pid = parton.id

        # Set hard limits on fragmentation values
        z_min = 0.01
        z_max = 1

        # Get a test array of z and P(z) values to compute max P
        z_val_array = np.arange(z_min, z_max, 0.01)
        p_z_array = []
        for z_val in z_val_array:
            p_z_val = self.pz(parton_pid, z_val, parton_pT)
            p_z_array.append(p_z_val)
        max_p_z = np.amax(p_z_array)

        # Sample a z value from the p(z) distribution
        z_val = np.array([])
        for attempt in np.arange(0, 10):
            # Randomly generate many points
            batch = 10000
            z_guesses = self.rng.uniform(z_min, z_max, batch)
            y_guesses = self.rng.uniform(0, max_p_z, batch)

            # Accept those under the curve
            for i in np.arange(0, len(z_guesses)):
                if self.pz(parton_pid, z_guesses[i], parton_pT) >= y_guesses[i]:
                    z_val = np.append(z_val, z_guesses[i])
                    if len(z_val) == num:
                        break
            if len(z_val) == num:
                break

        # Return sampled z
        if len(z_val) == 1:
            return z_val[0]
        else:
            return z_val