import math
import unittest

from hard_particles import Particle


class TestPartonPropagation(unittest.TestCase):
    def test_construct_sets_mass_and_name(self) -> None:
        g = Particle(id=21, px=1.0, py=0.0, pz=0.0, tau=1.0)
        self.assertEqual(g.id, 21)
        self.assertTrue(isinstance(g.m, float))
        self.assertEqual(g.m, 0.0)
        self.assertTrue(isinstance(g.name, str))
        self.assertTrue(len(g.name) > 0)

    def test_prop_dtau_zero_no_change(self) -> None:
        p = Particle(id=21, px=1.0, py=2.0, pz=3.0, tau=1.0, x=4.0, y=5.0, etas=0.1)
        before = (p.tau, p.x, p.y, p.etas)
        p.prop(0.0)
        after = (p.tau, p.x, p.y, p.etas)
        self.assertEqual(before, after)

    def test_massless_moves_at_light_speed_transverse_when_pz_zero(self) -> None:
        tau0 = 1.0
        dtau = 0.02
        nsteps = 500
        total_dtau = dtau * nsteps

        # pz=0, etas=0 => rap=0 => eta_s should stay fixed
        px, py, pz = 3.0, 4.0, 0.0  # pT = 5
        p = Particle(id=21, px=px, py=py, pz=pz, tau=tau0, x=1.0, y=-2.0, etas=0.0)

        # In this implementation, v_x = px/mT, v_y = py/mT, so for m=0 and pz=0 => vT=1
        expected_vT = 1.0

        # Propagate parton
        for _ in range(nsteps):
            p.prop(dtau)

        # Check parton time advanced properly and etas is constant at zero
        self.assertAlmostEqual(p.tau, tau0 + total_dtau, places=12)
        self.assertAlmostEqual(p.etas, 0.0, places=12)

        # Distance traveled in transverse plane should be ~ total_dtau (speed of light)
        rT = math.hypot(p.x - p.x_0, p.y - p.y_0)
        self.assertAlmostEqual(rT, total_dtau, places=9)

    def test_massive_slower_than_light_transverse(self) -> None:
        tau0 = 1.0
        dtau = 0.02
        nsteps = 500
        total_dtau = dtau * nsteps

        px, py, pz = 3.0, 4.0, 0.0
        p = Particle(id=3, px=px, py=py, pz=pz, tau=tau0, x=0.0, y=0.0, etas=0.0)

        expected_vx = px / p.mT
        expected_vy = py / p.mT
        expected_vT = math.hypot(expected_vx, expected_vy)

        self.assertLess(expected_vT, 1.0)

        # Propagate
        for _ in range(nsteps):
            p.prop(dtau)

        # Get distance traveled in transverse plane
        rT = math.hypot(p.x, p.y)

        # Distance traveled is less than light-speed case
        self.assertLess(rT, total_dtau)
        self.assertAlmostEqual(p.tau, tau0 + total_dtau, places=12)

        # Did not drift off transverse plane
        self.assertAlmostEqual(p.etas, 0.0, places=12)

    def test_etas_evolves_toward_rapidity_when_pz_nonzero(self) -> None:
        """
        With pz != 0, rap is nonzero and eta_s should evolve according to:
            d(eta_s)/d(tau) = sinh(rap - eta_s) / tau

        For constant rap=y, the analytic solution is:
            tanh((y - eta_s(tau))/2) = tanh((y - eta_s0)/2) * (tau0 / tau)
        """
        tau0 = 1.0
        dtau = 1e-3
        nsteps = 20000
        total_dtau = dtau * nsteps
        tau1 = tau0 + total_dtau

        # Keep pT nonzero to avoid mT=0 in the transverse velocity calculation.
        p = Particle(id=21, px=1.0, py=0.0, pz=2.0, tau=tau0, x=0.0, y=0.0, etas=0.0)

        y = p.rap
        etas0 = p.etas

        for _ in range(nsteps):
            p.prop(dtau)

        # Analytic expectation
        u0 = y - etas0
        k = math.tanh(0.5 * u0) * (tau0 / tau1)

        # Guard against tiny numerical overshoots beyond (-1, 1)
        eps = 1e-15
        k = max(min(k, 1.0 - eps), -1.0 + eps)

        u1 = 2.0 * math.atanh(k)
        expected_etas1 = y - u1

        self.assertAlmostEqual(p.tau, tau1, places=6)
        self.assertAlmostEqual(p.etas, expected_etas1, places=3)

        # Qualitative behavior: eta_s increases toward rap when starting below it
        self.assertGreater(p.etas, etas0)
        self.assertLess(p.etas, y)


if __name__ == "__main__":
    unittest.main()
