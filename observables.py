import numpy as np
import timeit

import matplotlib.pyplot as plt
import fastjet


def EEC(event, plot=False, pT_min=5, bins=np.linspace(-1, 1, 21)):
    """
    Function to compute energy-energy correlator for all particles in the event record.
    """

    # Access final state particles
    particles = [p for p in event if p.isFinal() and p.isCharged()]

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
        if p1.e() < pT_min:
            continue
        for j, p2 in enumerate(particles):
            # Enforce an energy cut on the second particle
            if p2.e() < pT_min:
                continue
            if i >= j: continue  # Avoid double counting

            # Calculate angle and energy weight -- use absolute value to set theta_2 > theta_1 -- exchanges i and j
            cos_theta = np.abs((p1.px() * p2.px() + p1.py() * p2.py() + p1.pz() * p2.pz()) / (p1.pAbs() * p2.pAbs()))
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