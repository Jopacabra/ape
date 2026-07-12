import pyhepmc
import os
import sys
import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import fastjet
sys.path.insert(0,'..')
import observables

# results subdirectory for HepMC files
name = "hepmc"

# Statistics settings
n_bootstrap_samples = 1000

# Jet finding particle cuts
R = 1.0
pTmin_jet_finder = 0.0
rap_min_jet_finder = 0.0
rap_max_jet_finder = 100.0

# Jet cuts
jet_minpt = 10.0
jet_maxpt = 100.0
rap_min_jet_axis = 0.0
rap_max_jet_axis = 0.1
phi_fence_jet_axis = 0.1

# Particle cuts
pTmin_vn = 2.0
pTmax_vn = 20
num_pt_bins = 8

# Flow attractor
v_jet_pts = []
v_jet_ys = []
v_jet_phis = []
m_jet_pts = []
m_jet_ys = []
m_jet_phis = []
flow = "x"
flow_i = 0
ls = ["-", "--", ".-", ":"]
color = ["r", "g", "b", "m"]

# define weighted mean stat for bootstrapping -- with axis setting for trickery with multi-dimensional data.
def weighted_mean(vn, w, axis=-1):
    return np.nansum(vn * w, axis=axis) / np.nansum(w, axis=axis)

# Iterate over cases
for case in ["v", "m"]:

    # Files
    hepmc_dir = "../results/" + name + "/" + case + "/"
    hepmc_files = os.listdir(hepmc_dir)


    # Binning & lists
    E_bins = np.linspace(pTmin_vn, pTmax_vn, num_pt_bins+1)
    v_weights = []
    v1s = []
    v2s = []
    v3s = []

    # Detail some info
    print(f"Number of {case} files: {len(hepmc_files)}")

    # Iterate over files and compute observables
    num_jets = 0
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    g = 0
    failed = 0
    analyzed_jets = []
    analyzed_weights = []
    for file in hepmc_files:
        try:
            with pyhepmc.open(hepmc_dir + file) as f:
                event = f.read()

            # Run jetfinder on this file
            try:
                jets = observables.hepmc_to_fastjet(hepmc_event=event, R=R, rap_min=rap_min_jet_finder, rap_max=rap_max_jet_finder, pTmin=pTmin_jet_finder,
                                                    scheme=fastjet.WTA_pt_scheme)
            except:
                print("failed file!")
                failed += 1
                continue

            # Get weight of event
            weight = event.weight("pythia")

            # Cut jet properties
            for i, jet in enumerate(jets):

                # pT cut
                jet_pt = jet.pt()
                if jet_pt < jet_minpt:
                    e += 1
                    continue
                elif jet_pt > jet_maxpt:
                    g += 1
                    continue

                # Perform phi cuts
                jet_p = np.array([jet.px(), jet.py(), jet.pz()])
                jet_phi = np.arctan2(jet_p[1], jet_p[0])
                if np.mod(jet_phi, np.pi/2) < phi_fence_jet_axis/2:
                    c += 1
                    continue
                elif np.mod(jet_phi, np.pi/2) > (np.pi/2 - phi_fence_jet_axis/2):
                    d += 1
                    continue

                # Perform rap cut
                jet_e = jet.e()
                jet_rap = 0.5 * np.log((jet_e + jet_p[2]) / (jet_e - jet_p[2]))
                if abs(jet_rap) < rap_min_jet_axis:
                    a += 1
                    continue
                elif abs(jet_rap) > rap_max_jet_axis:
                    b += 1
                    continue

                # Add to accepted jets
                analyzed_jets.append(jet)
                analyzed_weights.append(weight)

                # Record accepted values
                if case == "v":
                    v_jet_pts.append(jet_pt)
                    v_jet_phis.append(jet_phi)
                    v_jet_ys.append(jet_rap)
                elif case == "m":
                    m_jet_pts.append(jet_pt)
                    m_jet_phis.append(jet_phi)
                    m_jet_ys.append(jet_rap)

        except IsADirectoryError:
            continue

    # Readout
    print(f"Number of failed {case} events: {failed}")
    print(f"Number of too small pT {case} jets: {e}")
    print(f"Number of too large pT {case} jets: {g}")
    print(f"Number of too small rap {case} jets: {a}")
    print(f"Number of too large rap {case} jets: {b}")
    print(f"Number of too small phi {case} jets: {c}")
    print(f"Number of too large phi {case} jets: {d}")
    print(f"Number of just right {case} jets: {len(analyzed_jets)}")

    # Compute observable for accepted jets
    for i, jet in enumerate(analyzed_jets):

        # v1
        current_v1s, counts, _ = observables.fastjet_intrajetvnish_flow(jet=jet, pT_min=pTmin_vn, flow=flow, E_bins=E_bins, n=1)
        v1s.append(np.array(current_v1s, dtype=np.float32))

        # v2
        current_v2s, _, _ = observables.fastjet_intrajetvnish_flow(jet=jet, pT_min=pTmin_vn, flow=flow, E_bins=E_bins, n=2)
        v2s.append(np.array(current_v2s, dtype=np.float32))

        # v3
        current_v3s, _, _ = observables.fastjet_intrajetvnish_flow(jet=jet, pT_min=pTmin_vn, flow=flow, E_bins=E_bins, n=3)
        v3s.append(np.array(current_v3s, dtype=np.float32))
        num_jets += 1

        # weights
        v_weights.append(np.array(counts*analyzed_weights[i], dtype=np.float32))  # concat new weights onto old weights

    # Package weights as numpy array
    v_weights = np.array(v_weights)

    if case == "v":
        rng = np.random.default_rng()

        v1_res = bootstrap((v1s, v_weights), weighted_mean, axis=0, paired=True,
                           confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
        v1_avg = np.mean(v1_res.bootstrap_distribution, axis=1)
        v1_avg_err = v1_res.standard_error
        del v1_res

        v2_res = bootstrap((v2s, v_weights), weighted_mean, axis=0, paired=True,
                           confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
        v2_avg = np.mean(v2_res.bootstrap_distribution, axis=1)
        v2_avg_err = v2_res.standard_error
        del v2_res

        v3_res = bootstrap((v3s, v_weights), weighted_mean, axis=0, paired=True,
                           confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
        v3_avg = np.mean(v3_res.bootstrap_distribution, axis=1)
        v3_avg_err = v3_res.standard_error
        del v3_res

    elif case == "m":
        #############################################
        # Note error propagation adds in quadrature #
        #############################################

        rng = np.random.default_rng()

        v1_res = bootstrap((v1s, v_weights), weighted_mean, axis=0, paired=True,
                           confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
        v1_avg = np.mean(v1_res.bootstrap_distribution, axis=1) - v1_avg
        v1_avg_err = np.sqrt(v1_res.standard_error ** 2 + v1_avg_err ** 2)
        del v1_res

        v2_res = bootstrap((v2s, v_weights), weighted_mean, axis=0, paired=True,
                           confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
        v2_avg = np.mean(v2_res.bootstrap_distribution, axis=1) - v2_avg
        v2_avg_err = np.sqrt(v2_res.standard_error ** 2 + v2_avg_err ** 2)
        del v2_res

        v3_res = bootstrap((v3s, v_weights), weighted_mean, axis=0, paired=True,
                           confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
        v3_avg = np.mean(v3_res.bootstrap_distribution, axis=1) - v3_avg
        v3_avg_err = np.sqrt(v3_res.standard_error ** 2 + v3_avg_err ** 2)
        del v3_res


# # Plot
# E_bin_cents = (E_bins[1:] + E_bins[0:-1]) / 2
# fig = plt.figure(figsize=(8, 6))
# axis = fig.add_subplot()
# x_offset = 0.05 * (E_bins[-1] - E_bins[-2])  # 5% of the bin width.
# offsets = [-x_offset, 0, x_offset]  # error offsets
# axis.errorbar(E_bin_cents + offsets[0], v1_avg, yerr=v1_avg_err, capsize=10, elinewidth=1, marker='o', linestyle=ls[flow_i], color=color[flow_i], label=flow + r' $\Delta v_1$')
# axis.errorbar(E_bin_cents + offsets[1], v2_avg, yerr=v2_avg_err, capsize=10, elinewidth=1, marker='x', linestyle='--', color=color[flow_i+1], label=flow + r' $\Delta v_2$')
# axis.errorbar(E_bin_cents + offsets[2], v3_avg, yerr=v3_avg_err, capsize=10, elinewidth=1, marker='^', linestyle=':', color=color[flow_i+2], label=flow + r' $\Delta v_3$')
#
#
# axis.set_xlabel('E (GeV)', fontsize=14)
# axis.set_ylabel('Change in Average Harmonic', fontsize=14)
# axis.legend(fontsize=12)
# # plt.set_ylim(0.0, 0.35)
# # plt.set_xscale("log")
# # plt.set_yscale("log")
# axis.grid(True, alpha=0.3)
# axis.set_title(f"R={R}, phi_fence={phi_fence_jet_axis}, rap_min={rap_min_jet_axis}, rap_max={rap_max_jet_axis}, jetpTmin={jet_minpt}, jetpTmax={jet_maxpt}")
# fig.savefig(f"intrajet_vn_all.png")

# Plot
E_bin_cents = (E_bins[1:] + E_bins[0:-1]) / 2
fig = plt.figure(figsize=(12/2, 7/2))
axis = fig.add_subplot()
axis.axhline(y=0, color='black', linewidth=1.5, linestyle='--', zorder=1, alpha=0.5)

# v1
axis.plot(E_bin_cents, v1_avg, marker='o', markersize=7, linestyle='-',
          color=color[flow_i], linewidth=2, label=flow + r' $\Delta v_1$', zorder=3)
axis.fill_between(E_bin_cents, v1_avg - v1_avg_err, v1_avg + v1_avg_err,
                  color=color[flow_i], alpha=0.25, zorder=2)

# v2
axis.plot(E_bin_cents, v2_avg, marker='x', markersize=8, linestyle='--',
          color=color[flow_i+1], linewidth=2, label=flow + r' $\Delta v_2$', zorder=3)
axis.fill_between(E_bin_cents, v2_avg - v2_avg_err, v2_avg + v2_avg_err,
                  color=color[flow_i+1], alpha=0.25, zorder=2)

# v3
axis.plot(E_bin_cents, v3_avg, marker='^', markersize=7, linestyle=':',
          color=color[flow_i+2], linewidth=2, label=flow + r' $\Delta v_3$', zorder=3)
axis.fill_between(E_bin_cents, v3_avg - v3_avg_err, v3_avg + v3_avg_err,
                  color=color[flow_i+2], alpha=0.25, zorder=2)

axis.set_xlabel(r'$p_T$ (GeV)', fontsize=10)
axis.set_ylabel('Change in Average Harmonic', fontsize=10)
axis.legend(fontsize=10, loc='best')
axis.grid(True, alpha=0.3)
axis.set_title(f"R={R}, phi_fence={phi_fence_jet_axis}, particle_rap_min={rap_min_jet_finder}, particle_rap_max={rap_max_jet_finder},\njet_rap_min={rap_min_jet_axis}, jet_rap_max={rap_max_jet_axis}, jetpTmin={jet_minpt}, jetpTmax={jet_maxpt}", fontsize=10)
fig.savefig(f"intrajet_vn_all.png", dpi=150, bbox_inches='tight')

fig2 = plt.figure(figsize=(8, 6))
axis2 = fig2.add_subplot()
axis2.hist(v_jet_ys, bins=100, color="g", alpha=0.5, label="pp")
axis2.hist(m_jet_ys, bins=100, color="r", alpha=0.5, label="AA")
axis2.set_xlabel(r"$y_r$")
axis2.legend()
fig2.savefig("vn_rap_hist.png", dpi=150, bbox_inches="tight", pad_inches=0.05)

fig3 = plt.figure(figsize=(8, 6))
axis3 = fig3.add_subplot()
axis3.hist(v_jet_phis, bins=100, color="g", alpha=0.5, label="pp")
axis3.hist(m_jet_phis, bins=100, color="r", alpha=0.5, label="AA")
axis3.set_xlabel(r"$\phi$")
axis3.legend()
fig3.savefig("vn_phi_hist.png", dpi=150, bbox_inches="tight", pad_inches=0.05)

fig4 = plt.figure(figsize=(8, 6))
axis4 = fig4.add_subplot()
axis4.hist(v_jet_pts, bins=100, color="g", alpha=0.5, label="pp")
axis4.hist(m_jet_pts, bins=100, color="r", alpha=0.5, label="AA")
axis4.set_xlabel(r"jet $p_T$")
axis4.legend()
fig4.savefig("vn_pT_hist.png", dpi=150, bbox_inches="tight", pad_inches=0.05)

