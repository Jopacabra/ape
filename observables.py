import numpy as np
import timeit
import logging

import matplotlib.pyplot as plt
import fastjet
import pyhepmc
import pythia8

import utilities


def EEC(jet: fastjet.PseudoJet=None, event: pythia8.Event=None, plot=False, pT_min=1, bins=np.linspace(-1, 1, 21)):
    """
    Function to compute energy-energy correlator for all particles in the event record or in a jet
    """

    # Access final state particles
    if event is not None:
        particles = [p for p in event if p.isFinal() and p.isCharged()]
    elif jet is not None:
        particles = [p for p in jet.constituents()]
    else:
        raise ValueError("Either event or fastjet_jet must be provided.")

    # Compute total energy
    total_energy = 0
    for i, p1 in enumerate(particles):
        total_energy += p1.e()

    # Conceptual EEC calculation
    cos_array = np.array([])
    EEC_array = np.array([])
    weight_array = np.array([])
    for i, p1 in enumerate(particles):
        # Enforce an energy cut on the first particle
        if p1.pt() < pT_min:
            continue
        for j, p2 in enumerate(particles):
            # Enforce an energy cut on the second particle
            if p2.pt() < pT_min:
                continue
            if i >= j: continue  # Avoid double counting

            # Calculate angle and energy weight -- use absolute value to set theta_2 > theta_1 -- exchanges i and j
            magnitudes = (np.sqrt((p1.px() ** 2) + (p1.py() ** 2) + (p1.pz() ** 2))
                          * np.sqrt((p2.px() ** 2) + (p2.py() ** 2) + (p2.pz() ** 2)))
            cos_theta = np.abs((p1.px() * p2.px() + p1.py() * p2.py() + p1.pz() * p2.pz()) / magnitudes)
            weight = (p1.e() * p2.e()) / (total_energy ** 2)
            EEC = cos_theta * weight

            # Add to arrays
            cos_array = np.append(cos_array, cos_theta)
            EEC_array = np.append(EEC_array, EEC)
            weight_array = np.append(weight_array, weight)

    # Bin the values of cos_array and compute average EEC
    bin_centers = (bins[:-1] + bins[1:])/2  # Compute bin centers

    # Calculate weighted sums and average
    hist_values, _ = np.histogram(cos_array, bins=bins, weights=weight_array)
    hist_weighted_EEC, _ = np.histogram(cos_array, bins=bins, weights=EEC_array * weight_array)
    average_EEC = np.nan_to_num(hist_weighted_EEC / hist_values, nan=0.0)  # Sets nans from hist_values=0 to be 0.0.

    if plot:
        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, average_EEC, marker='o', linestyle='-', color='b', label='Weighted Avg EEC')
        plt.xlabel('cos(theta)', fontsize=14)
        plt.ylabel('Average EEC', fontsize=14)
        plt.title('Energy-Energy Correlation (EEC) vs. cos(theta)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return average_EEC, bins

def hepmc_to_fastjet(hepmc_event: pyhepmc.GenEvent, R: float=0.4, rap_min: float=0.0, rap_max: float=1.5, pTmin: float=0.0):
    """
    Function to run jetfinder on a HepMC event
    """
    # Access numpy interface of event object
    particles = hepmc_event.numpy.particles

    # Compute filter quantities
    E = particles.e
    px = particles.px
    py = particles.py
    pz = particles.pz
    denom = E - pz
    rap = 0.5 * np.log((E + pz) / denom)
    pT = np.hypot(px, py)

    # Filter
    ma = particles.status == 1  # Only consider final state particles
    # ma &= np.abs(particles.pid) == 211  # Only consider charged pions
    ma &= np.abs(rap) <= rap_max  # Cut on rapidity
    if rap_min > 0.0:
        ma &= np.abs(rap) > rap_min  # Cut on rapidity
    ma &= pT > pTmin  # Cut on transverse momentum

    # Iterate and make PseudoJets for fastjet.
    px = px[ma]
    py = py[ma]
    pz = pz[ma]
    E = E[ma]
    pseudo_particles = []
    for i in range(len(px)):
        # Append pseudojet for this little guy
        pseudo_particles.append(fastjet.PseudoJet(px[i], py[i], pz[i], E[i]))  # px, py, pz, E

    # Jet algorithm definition
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    logging.info("Jetfinding using FastJet algorithm: {}".format(jet_def))

    # Find jets
    jets = jet_def(pseudo_particles)

    return jets

def fastjet_intrajetvnish_flow(jet: fastjet.PseudoJet=None, event: pythia8.Event=None, n=1, pT_min=5, flow="x",
                          E_bins=None, phi_fence=0.2, rap_min=0.0, rap_max=1.5):
    """
    Function to compute intrajet v_n harmonic with reference angle set to select on either x or z flow.
    """

    # Access final state particles
    if event is not None:
        particles = [p for p in event if p.isFinal() and p.isCharged()]
    elif jet is not None:
        particles = [p for p in jet.constituents()]
    else:
        raise ValueError("Either event or fastjet_jet must be provided.")

    # Get jet axis direction as unit vector
    jet_p = np.array([jet.px(), jet.py(), jet.pz()])
    jet_p = jet_p / np.linalg.norm(jet_p)
    jet_e = jet.e()
    jet_rap = 0.5 * np.log((jet_e + jet_p[2]) / (jet_e - jet_p[2]))
    jet_phi = np.arctan2(jet_p[1], jet_p[0])

    # Get lists for particle properties
    phase = []
    E_array = []

    # Compute observable
    if flow == "x":
        # Find the unit vector perpendicular to jet_p and z-axis.
        # alpha_0_vec = np.sign(jet_p[0]*jet_p[1])*np.cross(jet_p, np.array([0, 0, 1]))
        xHat = np.array([1,0,0])
        alpha_0_vec = np.sign(jet_p[0]) * (xHat - np.dot(xHat, jet_p) * jet_p)
    elif flow == "z":
        # Find the unit vector perpendicular to jet_p and x-axis.
        # alpha_0_vec = (-1) * np.sign(jet_p[2]*jet_p[1])*np.cross(jet_p, np.array([1, 0, 0]))
        zHat = np.array([0,0,1])
        alpha_0_vec = np.sign(jet_p[2])*(zHat - np.dot(zHat, jet_p)*jet_p)
    elif flow == "total":
        yHat = np.array([0,1,0])
        alpha_0_vec = np.sign(jet_p[1])*(jet_p[1]*jet_p - yHat)

    alpha_0_vec = alpha_0_vec / np.linalg.norm(alpha_0_vec)

    # Compute cosine(alpha) and E of each particle.

    for i, p1 in enumerate(particles):
        if p1.pt() < pT_min:
            continue
        p1_p = np.array([p1.px(), p1.py(), p1.pz()])  # Constituent particle direction
        p1_perp = utilities.perp_vec(p1_p, jet_p)  # Component of p1_p perpendicular to jet_p
        p1_perp = p1_perp / np.linalg.norm(p1_perp)  # Normalized unit vector
        alpha = np.arccos(np.dot(alpha_0_vec, p1_perp))  # \vec{a}\cdot\vec{b} = ab \cos(\alpha), a = b = 1
        phase.append(np.cos(n*alpha))  # \cos(n * \alpha) phase factor for each particle
        E_array.append(p1.e())

    # Bin according to particle energy
    if E_bins is None:
        E_bins = np.linspace(pT_min, 15, 10)
    E_counts, _ = np.histogram(E_array, bins=E_bins)
    phase_sum, _ = np.histogram(E_array, bins=E_bins, weights=phase)

    # Compute averages and return
    avg_vns = phase_sum / E_counts  # Includes nans

    return avg_vns, E_bins

def fastjet_intrajetvnish_harmonics_flow_total(jet: fastjet.PseudoJet=None, event: pythia8.Event=None, pT_min=5, flow="x"):
    """
    Function to compute intrajet v_n harmonic with reference angle set to select on transverse, longitudinal, or all flow.
    """

    # Access final state particles
    if event is not None:
        particles = [p for p in event if p.isFinal() and p.isCharged()]
    elif jet is not None:
        particles = [p for p in jet.constituents()]
    else:
        raise ValueError("Either event or fastjet_jet must be provided.")

    # Get jet axis direction as unit vector
    jet_p = np.array([jet.px(), jet.py(), jet.pz()])
    jet_p = jet_p / np.linalg.norm(jet_p)

    if flow == "x":
        # Find the unit vector perpendicular to jet_p and z-axis.
        alpha_0_vec = np.sign(jet_p[0]*jet_p[1])*np.cross(jet_p, np.array([0, 0, 1]))
    elif flow == "z":
        # Find the unit vector perpendicular to jet_p and x-axis.
        alpha_0_vec = (-1) * np.sign(jet_p[2]*jet_p[1])*np.cross(jet_p, np.array([1, 0, 0]))
    elif flow == "total":
        alpha_0_vec = np.sign(jet_p[1])*(jet_p[1]*jet_p - np.array([0, 1, 0]))

    alpha_0_vec = alpha_0_vec / np.linalg.norm(alpha_0_vec)

    # Compute cosine(alpha) and E of each particle.
    phase_1 = []
    phase_2 = []
    phase_3 = []
    phase_4 = []
    for i, p1 in enumerate(particles):
        if p1.pt() < pT_min:
            continue
        p1_p = np.array([p1.px(), p1.py(), p1.pz()])  # Constituent particle direction
        p1_perp = utilities.perp_vec(p1_p, jet_p)  # Component of p1_p perpendicular to jet_p
        p1_perp = p1_perp / np.linalg.norm(p1_perp)  # Normalized unit vector
        alpha = np.arccos(np.dot(alpha_0_vec, p1_perp))  # \vec{a}\cdot\vec{b} = ab \cos(\alpha), a = b = 1
        phase_1.append(np.cos(1 * alpha))  # \cos(n * \alpha) phase factor for each particle
        phase_2.append(np.cos(2 * alpha))  # \cos(n * \alpha) phase factor for each particle
        phase_3.append(np.cos(3 * alpha))  # \cos(n * \alpha) phase factor for each particle
        phase_4.append(np.cos(4 * alpha))  # \cos(n * \alpha) phase factor for each particle

    # Bin according to particle energy
    mean_phase_1 = np.mean(phase_1)
    mean_phase_2 = np.mean(phase_2)
    mean_phase_3 = np.mean(phase_3)
    mean_phase_4 = np.mean(phase_4)

    return mean_phase_1, mean_phase_2, mean_phase_3, mean_phase_4
