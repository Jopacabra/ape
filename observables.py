import numpy as np
import timeit

import matplotlib.pyplot as plt
import fastjet
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

def fastjet_example():
    particles = []
    particles.append(fastjet.PseudoJet(100.0, 0.0, 0.0, 100.0))  # px, py, pz, E
    particles.append(fastjet.PseudoJet(150.0, 0.0, 0.0, 150.0))
    R = 0.4
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    jets = jet_def(particles)
    print(jet_def)
    for jet in jets: print(jet)

def fastjet_intrajetvnish(jet: fastjet.PseudoJet=None, event: pythia8.Event=None, n=1, pT_min=5, alpha_0=None,
                          E_bins=None):
    """
    Function to compute intrajet v_n harmonic using azimuthal angle about the jet axis.
    """

    # Access final state particles
    if event is not None:
        particles = [p for p in event if p.isFinal() and p.isCharged()]
    elif jet is not None:
        particles = [p for p in jet.constituents()]
    else:
        raise ValueError("Either event or fastjet_jet must be provided.")

    # Get jet axis direction
    jet_p = np.array([jet.px(), jet.py(), jet.pz()])

    # Defines the direction of zero azimuthal angle about jet axis (alpha = 0)
    if alpha_0 is None:  # Pick a random direction
        rng = np.random.default_rng()
        ran_vec = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        ran_vec = ran_vec   # Normalize
        zero_alpha = utilities.perp_vec(ran_vec, jet_p)  # Component of ran_vec perpendicular to jet_p
        zero_alpha = zero_alpha / np.linalg.norm(zero_alpha)  # Normalized unit vector
    else:
        alpha_0_vec = np.array([np.cos(alpha_0), np.sin(alpha_0), 0])
        zero_alpha = utilities.perp_vec(alpha_0_vec, jet_p)  # Component of alpha_0_vec perpendicular to jet_p
        zero_alpha = zero_alpha / np.linalg.norm(zero_alpha)  # Normalized unit vector

    # Compute cosine(alpha) and E of each particle.
    phase = np.array([])
    E_array = np.array([])
    for i, p1 in enumerate(particles):
        if p1.pt() < pT_min:
            continue
        p1_p = np.array([p1.px(), p1.py(), p1.pz()])  # Constituent particle direction
        p1_perp = utilities.perp_vec(p1_p, jet_p)  # Component of p1_p perpendicular to jet_p
        p1_perp = p1_perp / np.linalg.norm(p1_perp)  # Normalized unit vector
        alpha = np.arccos(np.dot(zero_alpha, p1_perp))  # \vec{a}\cdot\vec{b} = ab \cos(\alpha), a = b = 1
        phase = np.append(phase, np.cos(n*alpha))  # \cos(n * \alpha) phase factor for each particle
        E_array = np.append(E_array, p1.e())

    # Bin according to particle energy
    if E_bins is None:
        np.linspace(pT_min, 15, 10)
    E_counts, _ = np.histogram(E_array, bins=E_bins)
    phase_sum, _ = np.histogram(E_array, bins=E_bins, weights=phase)
    # print(phase_sum, E_counts)
    if alpha_0 is None:
        # Without a reference angle, we should consider all values to be positive.
        avg_vns = np.abs(np.nan_to_num((phase_sum / E_counts), nan=0.0))  # Sets nans from E_counts=0 to be 0.0.
    else:
        # With a reference angle, we care if the anisotropy is positive or negative.
        avg_vns = np.nan_to_num((phase_sum / E_counts), nan=0.0)  # Sets nans from E_counts=0 to be 0.0.

    return avg_vns, E_bins


