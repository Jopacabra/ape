import numpy as np
import matplotlib.pyplot as plt
import hard_particles
import plasma
import plasma_interaction as pi
import unittest

class TestAnalyticEnergyLoss(unittest.TestCase):
    def test_quadraticish_in_brick(self):
        T = 0.4
        u = 0.7
        rmax = 7.5

        # Make a slab
        def temp_func(t, x, y, etas):
            return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, T, 0.0)
        def x_vel_func(t, x, y, etas):
            return np.where(np.sqrt(x ** 2 + y ** 2) < rmax, u, 0.0)
        def y_vel_func(t, x, y, etas):
            return 0.0
        def z_vel_func(t, x, y, etas):
            return 0.0

        plasma_object = plasma.functional_plasma_3_1D(temp_func=temp_func,
                                                      x_vel_func=x_vel_func,
                                                      y_vel_func=y_vel_func,
                                                      z_vel_func=z_vel_func,
                                                      name=None, resolution=25, xmax=1.5*rmax, time=1.5*rmax, tau0=0.5)

        # Make a hard particle at the origin
        particle = hard_particles.Particle(
                    id=21,
                    px=5,
                    py=5,
                    pz=0,
                    tau=1,  # if None, Particle will choose its default in __post_init__
                    x=0,
                    y=0,
                    etas=0,
                    scalein=0,
                    col=101,
                    acol=102,
                    tag=0,
                    status=1,
                    mother1=0,
                    mother2=0,
                    daughter1=0,
                    daughter2=0
                )

        # Compute analytic expected energy of emissions in one step:
        dtau = 0.1  # fm
        steps = 50
        taus = []
        E_gluons = []
        N_gluons = []
        for i in np.arange(0, steps):
            E_gluons.append(pi.E_gluons(particle=particle, medium=plasma_object, dtau=dtau))
            N_gluons.append(pi.N_gluons(particle=particle, medium=plasma_object, dtau=dtau))
            particle.prop(dtau=dtau)
            taus.append(particle.tau)

        # Cast to numpy arrays
        taus = np.array(taus)
        cumulative_energy = np.cumsum(E_gluons)
        cumulative_number = np.cumsum(N_gluons)

        # Fit a quadratic function to the cumulative energy loss data
        E_coefficients = np.polyfit(taus, cumulative_energy, 2)
        N_coefficients = np.polyfit(taus, cumulative_number, 2)


        # Assert the fit has some quadratic elements
        assert E_coefficients[0] > 0.1, \
            "Cumulative energy of emitted gluons does not follow an approximately quadratic trend."
        assert N_coefficients[0] > 0.1, \
            "Cumulative number of emitted gluons does not follow an approximately quadratic trend."

if __name__ == "__main__":
    unittest.main()