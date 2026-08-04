import pyhepmc
import os
import sys
import contextlib
import io
import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import fastjet
sys.path.insert(0,'..')
import observables

@contextlib.contextmanager
def suppress_fd_output():
    """Suppresses stdout and stderr at the OS file-descriptor level.
    Works even for output from compiled C/C++ extensions."""
    # Open null file descriptors
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    # Save the real stdout (1) and stderr (2) file descriptors
    saved_fds = [os.dup(1), os.dup(2)]
    try:
        # Redirect stdout and stderr to /dev/null
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        # Restore original stdout and stderr
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        # Close all temporary file descriptors
        for fd in null_fds + saved_fds:
            os.close(fd)

# results subdirectory for HepMC files
hepmc_dir = "../results_saved/0_10_avg_AuAu_post_finkin_fix/hepmc/"
label = "0-10%"

# Statistics settings
n_bootstrap_samples = 10000

# Jet finding cuts
R = 1.0
pTmin_jet_finder = 2.0  # Minimum pT of constituents to consider
rap_min_jet_finder = 0.0  # Minimum rapidity of constituents to consider
rap_max_jet_finder = R / 2   # Maximum rapidity of constituents to consider

# Jet cuts
jet_minpt = 10.0
jet_maxpt = 120.0
rap_min_jet_axis = 0.0
rap_max_jet_axis = 0.5
phi_fence_jet_axis = 0.4
fence_y = True

# Binning
cut_by = "x"
pTmin_vn = 10.0  # don't use zero if you log space bins.
pTmax_vn = 120.0
num_pt_bins = 5
spacing = "lin"  # spacing for pt bins only
xmin_vn = 0.2  # always lin spaced, if chosen
xmax_vn = 1
num_x_bins = 5

# Observables settings
flow = "x"
pt_weighting = 0  # Weight each particle's phase by p_T^(pt_weighting) when computing vn of each jet
refaxis = "jet"
plot_group = "diff"  # "diff", "both", "m", or "v"

# Storage arrays
v_jet_pts = []
v_jet_ys = []
v_jet_phis = []
m_jet_pts = []
m_jet_ys = []
m_jet_phis = []
bin_vals_v1 = []
bin_errs_v1 = []
bin_vals_v2 = []
bin_errs_v2 = []
bin_vals_v3 = []
bin_errs_v3 = []
v_bin_vals_v1 = []
v_bin_errs_v1 = []
v_bin_vals_v2 = []
v_bin_errs_v2 = []
v_bin_vals_v3 = []
v_bin_errs_v3 = []

# Plotting stuff
flow_i = 0
ls = ["-", "--", ".-", ":"]
color = ["r", "g", "b", "m"]

# define weighted mean stat for bootstrapping
def weighted_mean(vn, w):
    return np.nansum(vn * w) / np.nansum(w)

# loop over vacuum and medium cases
for case in ["v", "m"]:

    # Files
    case_dir = hepmc_dir + case + "/"
    hepmc_files = os.listdir(case_dir)


    # Binning & lists
    if cut_by == "pt":
        if spacing == "lin":
            E_bins = np.linspace(pTmin_vn, pTmax_vn, num_pt_bins + 1)
        elif spacing == "log":
            E_bins = np.logspace(np.log10(pTmin_vn), np.log10(pTmax_vn), num_pt_bins + 1)
    elif cut_by == "x":
        x_bins = np.linspace(xmin_vn, xmax_vn, num_x_bins + 1)
    v1s = []
    v2s = []
    v3s = []

    # Iterate over files and compute observables
    num_jets = 0
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    g = 0
    no_pair = 0
    failed = 0
    analyzed_jets = []
    analyzed_jet_pts = []
    analyzed_gammas = []
    analyzed_gamma_pts = []
    analyzed_weights = []
    for file in hepmc_files:
        try:
            with suppress_fd_output():
                with pyhepmc.open(case_dir + file) as f:
                    event = f.read()


            # Run jetfinder on this file
            try:
                gamma, jet = observables.hepmc_to_fastjet_gamma_jet_pairs(hepmc_event=event, R=R,
                                                                          rap_min=rap_min_jet_finder,
                                                                          rap_max=rap_max_jet_finder,
                                                                          pTmin=pTmin_jet_finder,
                                                                          scheme=fastjet.WTA_pt_scheme)
            except:
                # print("failed file!")
                failed += 1
                continue

            if jet is None:
                no_pair += 1
                continue

            # Get weight of event
            weight = event.weight("pythia")

            # jet pT cut
            jet_pt = jet.pt()
            if jet_pt < jet_minpt:
                e += 1
                continue
            elif jet_pt > jet_maxpt:
                g += 1
                continue

            # jet phi cuts
            jet_p = np.array([jet.px(), jet.py(), jet.pz()])
            jet_phi = np.arctan2(jet_p[1], jet_p[0])
            if np.mod(jet_phi, np.pi/2) < phi_fence_jet_axis/2:
                c += 1
                continue
            elif fence_y and np.mod(jet_phi, np.pi/2) > (np.pi/2 - phi_fence_jet_axis/2):
                d += 1
                continue

            # jet rap cut
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
            analyzed_gammas.append(gamma)
            analyzed_gamma_pts.append(gamma.pt())
            analyzed_jet_pts.append(jet_pt)
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
    print(f"Number of {case} files: {len(hepmc_files)}")
    print(f"Number of failed {case} events: {failed}")
    print(f"Number of no pair {case} events: {no_pair}")
    print(f"Number of too small pT {case} jets: {e}")
    print(f"Number of too large pT {case} jets: {g}")
    print(f"Number of too small rap {case} jets: {a}")
    print(f"Number of too large rap {case} jets: {b}")
    print(f"Number of too small phi {case} jets: {c}")
    print(f"Number of too large phi {case} jets: {d}")
    print(f"Number of just right {case} jets: {len(analyzed_jets)}")

    n_constituents = []
    for jet in analyzed_jets:
        parts = [p for p in jet.constituents()]
        n_constituents.append(len(parts))
    print(f"Mean constituents after cut: {np.mean(n_constituents):.2f}")
    print(f"Fraction with N<=2: {np.mean(np.array(n_constituents) <= 2):.2f}")

    # Compute observable for accepted jets
    for gamma, jet in zip(analyzed_gammas, analyzed_jets):

        # Compute vns
        if refaxis == "jet":
            current_v1s = observables.fastjet_intrajetvnish_total(jet=jet, flow=flow, n=1,
                                                                           pt_weighting=pt_weighting)
            current_v2s = observables.fastjet_intrajetvnish_total(jet=jet, flow=flow, n=2,
                                                                           pt_weighting=pt_weighting)
            current_v3s = observables.fastjet_intrajetvnish_total(jet=jet, flow=flow, n=3,
                                                                           pt_weighting=pt_weighting)
        elif refaxis == "gamma":
            current_v1s = observables.fastjet_intrajetvnish_total_gammaref(jet=jet, gamma=gamma, flow=flow, n=1,
                                                                           pt_weighting=pt_weighting)
            current_v2s = observables.fastjet_intrajetvnish_total_gammaref(jet=jet, gamma=gamma, flow=flow, n=2,
                                                                           pt_weighting=pt_weighting)
            current_v3s = observables.fastjet_intrajetvnish_total_gammaref(jet=jet, gamma=gamma, flow=flow, n=3,
                                                                           pt_weighting=pt_weighting)
        v1s.append(np.array(current_v1s, dtype=np.float32))
        v2s.append(np.array(current_v2s, dtype=np.float32))
        v3s.append(np.array(current_v3s, dtype=np.float32))
        num_jets += 1

    # Bin jets
    v1s = np.array(v1s)
    v2s = np.array(v2s)
    v3s = np.array(v3s)

    plt.scatter(n_constituents, v1s, alpha=0.1, s=1)
    N_theory = np.arange(1, 30)
    plt.plot(N_theory, 1 / np.sqrt(N_theory) * np.sqrt(np.pi / 4), 'r--', label=r'$\sqrt{\pi}/2\sqrt{N}$')
    plt.xlabel("N constituents")
    plt.ylabel(r"$|Q_1|$")
    plt.savefig(f"vn_vs_N_{case}.png")

    # Store for later histogramming
    if case == "v":
        v_analyzed_weights = np.array(analyzed_weights)
    elif case == "m":
        m_analyzed_weights = np.array(analyzed_weights)

    # Cast to numpy arrays
    print(np.nanmean(v1s))
    print(np.nanmax(v1s))
    print(np.nanmin(v1s))
    analyzed_weights = np.array(analyzed_weights)
    analyzed_gamma_pts = np.array(analyzed_gamma_pts)
    analyzed_jet_pts = np.array(analyzed_jet_pts)
    analyzed_x = analyzed_jet_pts / analyzed_gamma_pts
    if cut_by == "pt":
        cut_bins = E_bins
        analyzed_domain = analyzed_jet_pts
    elif cut_by == "x":
        cut_bins = x_bins
        analyzed_domain = analyzed_x
    for j, cut_min in enumerate(cut_bins[0:-1]):
        # pt_counts, _ = np.histogram(analyzed_jet_pts, bins=E_bins)
        # v1s_binned, _ = np.histogram(analyzed_jet_pts, bins=E_bins, weights=v1s)
        cut_max = cut_bins[j+1]

        cut = (analyzed_domain > cut_min) & (analyzed_domain < cut_max)
        v1s_bin = v1s[cut]
        v2s_bin = v2s[cut]
        v3s_bin = v3s[cut]
        v_weights = analyzed_weights[cut]

        if len(v_weights) < 2:  # Fill empty for nans
            if case == "v":
                v_bin_vals_v1.append(np.nan)
                v_bin_errs_v1.append(np.nan)
                v_bin_vals_v2.append(np.nan)
                v_bin_errs_v2.append(np.nan)
                v_bin_vals_v3.append(np.nan)
                v_bin_errs_v3.append(np.nan)
            elif case == "m":
                bin_vals_v1.append(np.nan)
                bin_errs_v1.append(np.nan)
                bin_vals_v2.append(np.nan)
                bin_errs_v2.append(np.nan)
                bin_vals_v3.append(np.nan)
                bin_errs_v3.append(np.nan)
            continue

        else:
            # Compute vns in this bin using bootstrap distribution
            rng = np.random.default_rng()
            v1_data = (v1s_bin, v_weights)
            v1_res = bootstrap(v1_data, weighted_mean,
                               confidence_level=0.9, rng=rng,
                               n_resamples=n_bootstrap_samples,
                               paired=True)  # <-- critical: keeps vn[i] paired with w[i]
            # v1_avg = np.mean(v1_res.bootstrap_distribution)
            v1_avg = weighted_mean(v1s_bin, v_weights)
            v1_avg_err = v1_res.standard_error

            v2_data = (v2s_bin, v_weights)
            v2_res = bootstrap(v2_data, weighted_mean,
                               confidence_level=0.9, rng=rng,
                               n_resamples=n_bootstrap_samples,
                               paired=True)  # <-- critical: keeps vn[i] paired with w[i]
            # v2_avg = np.mean(v2_res.bootstrap_distribution)
            v2_avg = weighted_mean(v2s_bin, v_weights)
            v2_avg_err = v2_res.standard_error

            v3_data = (v3s_bin, v_weights)
            v3_res = bootstrap(v3_data, weighted_mean,
                               confidence_level=0.9, rng=rng,
                               n_resamples=n_bootstrap_samples,
                               paired=True)  # <-- critical: keeps vn[i] paired with w[i]
            # v3_avg = np.mean(v3_res.bootstrap_distribution)
            v3_avg = weighted_mean(v3s_bin, v_weights)
            v3_avg_err = v3_res.standard_error

            # Append to appropriate list
            if case == "v":
                v_bin_vals_v1.append(v1_avg)
                v_bin_errs_v1.append(v1_avg_err)
                v_bin_vals_v2.append(v2_avg)
                v_bin_errs_v2.append(v2_avg_err)
                v_bin_vals_v3.append(v3_avg)
                v_bin_errs_v3.append(v3_avg_err)

            elif case == "m":
                bin_vals_v1.append(v1_avg)
                bin_errs_v1.append(v1_avg_err)
                bin_vals_v2.append(v2_avg)
                bin_errs_v2.append(v2_avg_err)
                bin_vals_v3.append(v3_avg)
                bin_errs_v3.append(v3_avg_err)

# Make numpy arrays
bin_vals_v1 = np.array(bin_vals_v1)
bin_errs_v1 = np.array(bin_errs_v1)
bin_vals_v2 = np.array(bin_vals_v2)
bin_errs_v2 = np.array(bin_errs_v2)
bin_vals_v3 = np.array(bin_vals_v3)
bin_errs_v3 = np.array(bin_errs_v3)
v_bin_vals_v1 = np.array(v_bin_vals_v1)
v_bin_errs_v1 = np.array(v_bin_errs_v1)
v_bin_vals_v2 = np.array(v_bin_vals_v2)
v_bin_errs_v2 = np.array(v_bin_errs_v2)
v_bin_vals_v3 = np.array(v_bin_vals_v3)
v_bin_errs_v3 = np.array(v_bin_errs_v3)

print(v_bin_vals_v1)
print(bin_vals_v1)
print(v_bin_vals_v2)
print(bin_vals_v2)
print(v_bin_vals_v3)
print(bin_vals_v3)

# Plot
cut_bin_cents = (cut_bins[1:] + cut_bins[0:-1]) / 2
fig = plt.figure(figsize=(12/2, 7/2))
axis = fig.add_subplot()
axis.axhline(y=0, color='black', linewidth=1.5, linestyle='--', zorder=1, alpha=0.5)

if plot_group == "diff":
    # Take the difference
    diff_bin_vals_v1 = bin_vals_v1 - v_bin_vals_v1
    diff_bin_vals_v2 = bin_vals_v2 - v_bin_vals_v2
    diff_bin_vals_v3 = bin_vals_v3 - v_bin_vals_v3

    # Add errors in quadrature
    diff_bin_errs_v1 = np.sqrt(v_bin_errs_v1**2 + bin_errs_v1**2)
    diff_bin_errs_v2 = np.sqrt(v_bin_errs_v2**2 + bin_errs_v2**2)
    diff_bin_errs_v3 = np.sqrt(v_bin_errs_v3**2 + bin_errs_v3**2)

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



    # v1
    axis.plot(cut_bin_cents, diff_bin_vals_v1, marker='o', markersize=7, linestyle='-',
              color=color[flow_i], linewidth=2, label=flow + r' $\Delta v_1$', zorder=3)
    axis.fill_between(cut_bin_cents, diff_bin_vals_v1 - diff_bin_errs_v1, diff_bin_vals_v1 + diff_bin_errs_v1,
                      color=color[flow_i], alpha=0.25, zorder=2)

    # v2
    axis.plot(cut_bin_cents, diff_bin_vals_v2, marker='x', markersize=8, linestyle='--',
              color=color[flow_i+1], linewidth=2, label=flow + r' $\Delta v_2$', zorder=3)
    axis.fill_between(cut_bin_cents, diff_bin_vals_v2 - diff_bin_errs_v2, diff_bin_vals_v2 + diff_bin_errs_v2,
                      color=color[flow_i+1], alpha=0.25, zorder=2)

    # v3
    axis.plot(cut_bin_cents, diff_bin_vals_v3, marker='^', markersize=7, linestyle=':',
              color=color[flow_i+2], linewidth=2, label=flow + r' $\Delta v_3$', zorder=3)
    axis.fill_between(cut_bin_cents, diff_bin_vals_v3 - diff_bin_errs_v3, diff_bin_vals_v3 + diff_bin_errs_v3,
                      color=color[flow_i+2], alpha=0.25, zorder=2)

    axis.set_ylabel('Change in Average Harmonic', fontsize=10)
elif plot_group == "m":
    # v1 -- AA
    axis.plot(cut_bin_cents, bin_vals_v1, marker='o', markersize=7, linestyle='-',
              color=color[flow_i], linewidth=2, label=flow + r' AA $v_1$', zorder=3)
    axis.fill_between(cut_bin_cents, bin_vals_v1 - bin_errs_v1, bin_vals_v1 + bin_errs_v1,
                      color=color[flow_i], alpha=0.25, zorder=2)

    # v2 -- AA
    axis.plot(cut_bin_cents, bin_vals_v2, marker='x', markersize=8, linestyle='--',
              color=color[flow_i + 1], linewidth=2, label=flow + r' AA $v_2$', zorder=3)
    axis.fill_between(cut_bin_cents, bin_vals_v2 - bin_errs_v2, bin_vals_v2 + bin_errs_v2,
                      color=color[flow_i + 1], alpha=0.25, zorder=2)

    # v3 -- AA
    axis.plot(cut_bin_cents, bin_vals_v3, marker='^', markersize=7, linestyle=':',
              color=color[flow_i + 2], linewidth=2, label=flow + r' AA $v_3$', zorder=3)
    axis.fill_between(cut_bin_cents, bin_vals_v3 - bin_errs_v3, bin_vals_v3 + bin_errs_v3,
                      color=color[flow_i + 2], alpha=0.25, zorder=2)

    axis.set_ylabel('Average Harmonic', fontsize=10)
elif plot_group == "v":
    # v1 -- pp
    axis.plot(cut_bin_cents, v_bin_vals_v1, marker='o', markersize=7, linestyle='-',
              color=color[flow_i], linewidth=2, label=flow + r' pp $v_1$', zorder=3)
    axis.fill_between(cut_bin_cents, v_bin_vals_v1 - v_bin_errs_v1, v_bin_vals_v1 + v_bin_errs_v1,
                      color=color[flow_i], alpha=0.25, zorder=2)

    # v2 -- pp
    axis.plot(cut_bin_cents, v_bin_vals_v2, marker='x', markersize=8, linestyle='--',
              color=color[flow_i + 1], linewidth=2, label=flow + r' pp $v_2$', zorder=3)
    axis.fill_between(cut_bin_cents, v_bin_vals_v2 - v_bin_errs_v2, v_bin_vals_v2 + v_bin_errs_v2,
                      color=color[flow_i + 1], alpha=0.25, zorder=2)

    # v3 -- pp
    axis.plot(cut_bin_cents, v_bin_vals_v3, marker='^', markersize=7, linestyle=':',
              color=color[flow_i + 2], linewidth=2, label=flow + r' pp $v_3$', zorder=3)
    axis.fill_between(cut_bin_cents, v_bin_vals_v3 - v_bin_errs_v3, bin_vals_v3 + v_bin_errs_v3,
                      color=color[flow_i + 2], alpha=0.25, zorder=2)

    axis.set_ylabel('Average Harmonic', fontsize=10)
elif plot_group == "both":
    # v1 -- pp
    axis.plot(cut_bin_cents, v_bin_vals_v1, marker='o', markersize=7, linestyle=':',
              color=color[flow_i], linewidth=2, label=flow + r' pp $v_1$', zorder=3)
    axis.fill_between(cut_bin_cents, v_bin_vals_v1 - v_bin_errs_v1, v_bin_vals_v1 + v_bin_errs_v1,
                      color=color[flow_i], alpha=0.25, zorder=2)

    # v2 -- pp
    axis.plot(cut_bin_cents, v_bin_vals_v2, marker='x', markersize=8, linestyle=':',
              color=color[flow_i + 1], linewidth=2, label=flow + r' pp $v_2$', zorder=3)
    axis.fill_between(cut_bin_cents, v_bin_vals_v2 - v_bin_errs_v2, v_bin_vals_v2 + v_bin_errs_v2,
                      color=color[flow_i + 1], alpha=0.25, zorder=2)

    # v3 -- pp
    axis.plot(cut_bin_cents, v_bin_vals_v3, marker='^', markersize=7, linestyle=':',
              color=color[flow_i + 2], linewidth=2, label=flow + r' pp $v_3$', zorder=3)
    axis.fill_between(cut_bin_cents, v_bin_vals_v3 - v_bin_errs_v3, bin_vals_v3 + v_bin_errs_v3,
                      color=color[flow_i + 2], alpha=0.25, zorder=2)


    # v1 -- AA
    axis.plot(cut_bin_cents, bin_vals_v1, marker='o', markersize=7, linestyle='-',
              color=color[flow_i], linewidth=2, label=flow + r' AA $\Delta v_1$', zorder=3)
    axis.fill_between(cut_bin_cents, bin_vals_v1 - bin_errs_v1, bin_vals_v1 + bin_errs_v1,
                      color=color[flow_i], alpha=0.25, zorder=2)

    # v2 -- AA
    axis.plot(cut_bin_cents, bin_vals_v2, marker='x', markersize=8, linestyle='-',
              color=color[flow_i + 1], linewidth=2, label=flow + r' AA $\Delta v_2$', zorder=3)
    axis.fill_between(cut_bin_cents, bin_vals_v2 - bin_errs_v2, bin_vals_v2 + bin_errs_v2,
                      color=color[flow_i + 1], alpha=0.25, zorder=2)

    # v3 -- AA
    axis.plot(cut_bin_cents, bin_vals_v3, marker='^', markersize=7, linestyle='-',
              color=color[flow_i + 2], linewidth=2, label=flow + r' AA $\Delta v_3$', zorder=3)
    axis.fill_between(cut_bin_cents, bin_vals_v3 - bin_errs_v3, bin_vals_v3 + bin_errs_v3,
                      color=color[flow_i + 2], alpha=0.25, zorder=2)


    axis.set_ylabel('Average Harmonic', fontsize=10)

if cut_by == "x":
    axis.set_xlabel(r'$x_j$', fontsize=10)
else:
    axis.set_xlabel(r'Jet $p_T$ (GeV)', fontsize=10)
axis.legend(fontsize=10, loc='best')
axis.grid(True, alpha=0.3)
axis.set_title(f"{label} Cent., R={R}, phi_fence={phi_fence_jet_axis}, Fence_y={fence_y}, particle_rap_min={rap_min_jet_finder}, particle_rap_max={rap_max_jet_finder},\njet_rap_min={rap_min_jet_axis}, jet_rap_max={rap_max_jet_axis}, jetpTmin={jet_minpt}, jetpTmax={jet_maxpt}", fontsize=10)
money_fname = f"intrajet_vn_{refaxis}axis_jet{cut_by}.png"
fig.savefig(money_fname, dpi=150, bbox_inches='tight')
print(f"Saved to: {money_fname}")

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

bins = np.linspace(10, 120, 30)
fig4 = plt.figure(figsize=(8, 6))
axis4 = fig4.add_subplot()
axis4.hist(v_jet_pts, bins=bins, color="g", alpha=0.5, label="pp")
axis4.hist(m_jet_pts, bins=bins, color="r", alpha=0.5, label="AA")
axis4.set_xlabel(r"jet $p_T$")
axis4.set_ylabel(r"$N$ Statistical")
axis4.legend()
fig4.savefig("vn_pT_hist_stat.png", dpi=150, bbox_inches="tight", pad_inches=0.05)

fig5 = plt.figure(figsize=(8, 6))
axis5 = fig5.add_subplot()

axis5.hist(v_jet_pts, bins=bins, color="g", alpha=0.5, label="pp", weights=v_analyzed_weights)
axis5.hist(m_jet_pts, bins=bins, color="r", alpha=0.5, label="AA", weights=m_analyzed_weights)
axis5.set_xlabel(r"jet $p_T$")
axis5.set_ylabel(r"$N$ Physical")
axis5.legend()
fig5.savefig("vn_pT_hist_phys.png", dpi=150, bbox_inches="tight", pad_inches=0.05)

