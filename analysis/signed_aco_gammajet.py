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
name = "hepmc"#"hepmc_1_120_GeV"

# Statistics settings
n_bootstrap_samples = 10000

# Jet finding cuts
R = 1.0
pTmin_jet_finder = 2.0  # Minimum pT of constituents to consider
rap_min_jet_finder = 0.0  # Minimum rapidity of constituents to consider
rap_max_jet_finder = 1.5  # Maximum rapidity of constituents to consider

# Jet cuts
jet_minpt = 10.0
jet_maxpt = 120.0
rap_min_jet_axis = 0.0
rap_max_jet_axis = 0.5
phi_fence_jet_axis = 0.2
fence_y = True

# Binning
acomin = -np.pi/2 # don't use zero if you log space bins.
acomax = +np.pi/2
num_aco_bins = 32
aco_bins = np.linspace(acomin, acomax, num_aco_bins)

# Observables settings
dir = "phi"
pt_weighting = 0  # Weight each particle's phase by p_T^(pt_weighting) when computing vn of each jet
plot_group = "both"  # "ratio", "diff", "both", "m", or "v"

# Storage arrays
v_jet_pts = []
v_jet_ys = []
v_jet_phis = []
m_jet_pts = []
m_jet_ys = []
m_jet_phis = []
bin_vals_aco = []
bin_errs_aco = []
v_bin_vals_aco = []
v_bin_errs_aco = []

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
    hepmc_dir = "../results/" + name + "/" + case + "/"
    hepmc_files = os.listdir(hepmc_dir)


    # Binning & lists
    acos = []

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
                with pyhepmc.open(hepmc_dir + file) as f:
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

        # Compute aco
        current_aco = observables.signed_deflection(jet=jet, gamma=gamma, dir=dir)
        acos.append(np.array(current_aco, dtype=np.float32))
        num_jets += 1

    # Bin jets
    acos = np.array(acos)

    print(np.nanmean(acos))
    print(np.nanmax(acos))
    print(np.nanmin(acos))
    analyzed_weights = np.array(analyzed_weights)
    analyzed_gamma_pts = np.array(analyzed_gamma_pts)
    analyzed_jet_pts = np.array(analyzed_jet_pts)
    analyzed_x = analyzed_jet_pts / analyzed_gamma_pts

    for j, aco_min in enumerate(aco_bins[0:-1]):
        # pt_counts, _ = np.histogram(analyzed_jet_pts, bins=E_bins)
        # v1s_binned, _ = np.histogram(analyzed_jet_pts, bins=E_bins, weights=v1s)
        aco_max = aco_bins[j+1]


        cut = (acos > aco_min) & (acos < aco_max)
        # acos_bin = acos[cut]
        v_weights = analyzed_weights[cut]
        print(len(v_weights))

        if len(v_weights) < 2:  # Fill empty for nans
            if case == "v":
                v_bin_vals_aco.append(np.nan)
                v_bin_errs_aco.append(np.nan)
            elif case == "m":
                bin_vals_aco.append(np.nan)
                bin_errs_aco.append(np.nan)
            continue

        else:
            # Compute vns in this bin using bootstrap distribution
            rng = np.random.default_rng()
            aco_data = (v_weights,)
            aco_res = bootstrap(aco_data, np.nansum,
                               confidence_level=0.9, rng=rng,
                               n_resamples=n_bootstrap_samples,
                               paired=True)  # <-- critical: keeps vn[i] paired with w[i]
            # v1_avg = np.mean(v1_res.bootstrap_distribution)
            aco_num = np.nansum(v_weights)
            aco_num_err = aco_res.standard_error

            # Append to appropriate list
            if case == "v":
                v_bin_vals_aco.append(aco_num)
                v_bin_errs_aco.append(aco_num_err)

            elif case == "m":
                bin_vals_aco.append(aco_num)
                bin_errs_aco.append(aco_num_err)

# Make numpy arrays
bin_vals_aco = np.array(bin_vals_aco)
bin_errs_aco = np.array(bin_errs_aco)
v_bin_vals_aco = np.array(v_bin_vals_aco)
v_bin_errs_aco = np.array(v_bin_errs_aco)

print(v_bin_vals_aco)
print(bin_vals_aco)

# Plot
cut_bin_cents = (aco_bins[1:] + aco_bins[0:-1]) / 2
fig = plt.figure(figsize=(12/2, 7/2))
axis = fig.add_subplot()
axis.axhline(y=0, color='black', linewidth=1.5, linestyle='--', zorder=1, alpha=0.5)

if plot_group == "ratio":
    # Take ratio
    ratio_bin_vals_aco = bin_vals_aco / v_bin_vals_aco

    # Add errors appropriately
    ratio_bin_errs_aco = np.zeros_like(ratio_bin_vals_aco)

    # Ratio plot
    axis.plot(cut_bin_cents, ratio_bin_vals_aco, marker='o', markersize=7, linestyle='-',
              color=color[flow_i], linewidth=2, label=dir + r' $\Delta aco$', zorder=3)
    axis.fill_between(cut_bin_cents, ratio_bin_vals_aco - ratio_bin_errs_aco, ratio_bin_vals_aco + ratio_bin_errs_aco,
                      color=color[3], alpha=0.25, zorder=2)

    axis.set_ylabel(r'$R_{\phi_{\psi_2}}$', fontsize=10)

elif plot_group == "diff":
    # Take the difference
    diff_bin_vals_aco = bin_vals_aco - v_bin_vals_aco

    # Add errors in quadrature
    diff_bin_errs_aco = np.sqrt(v_bin_errs_aco**2 + bin_errs_aco**2)

    # aco diff
    axis.plot(cut_bin_cents, diff_bin_vals_aco, marker='o', markersize=7, linestyle='-',
              color=color[flow_i], linewidth=2, label=dir + r' $\Delta aco$', zorder=3)
    axis.fill_between(cut_bin_cents, diff_bin_vals_aco - diff_bin_errs_aco, diff_bin_vals_aco + diff_bin_errs_aco,
                      color=color[flow_i], alpha=0.25, zorder=2)

    axis.set_ylabel('Change in Average Signed Acoplanarity (rad)', fontsize=10)
elif plot_group == "m":
    # AA
    axis.plot(cut_bin_cents, bin_vals_aco, marker='o', markersize=7, linestyle='-',
              color=color[flow_i], linewidth=2, label=dir + r' AA $aco$', zorder=3)
    axis.fill_between(cut_bin_cents, bin_vals_aco - bin_errs_aco, bin_vals_aco + bin_errs_aco,
                      color=color[flow_i], alpha=0.25, zorder=2)

    axis.set_ylabel('Average Signed Acoplanarity (rad)', fontsize=10)
elif plot_group == "v":
    # pp
    axis.plot(cut_bin_cents, v_bin_vals_aco, marker='o', markersize=7, linestyle='-',
              color=color[1], linewidth=2, label=dir + r' pp $aco$', zorder=3)
    axis.fill_between(cut_bin_cents, v_bin_vals_aco - v_bin_errs_aco, v_bin_vals_aco + v_bin_errs_aco,
                      color=color[1], alpha=0.25, zorder=2)

    axis.set_ylabel('Average Signed Acoplanarity (rad)', fontsize=10)
elif plot_group == "both":
    # AA
    axis.plot(cut_bin_cents, bin_vals_aco, marker='o', markersize=7, linestyle='-',
              color=color[0], linewidth=2, label=dir + r' AA $aco$', zorder=3)
    axis.fill_between(cut_bin_cents, bin_vals_aco - bin_errs_aco, bin_vals_aco + bin_errs_aco,
                      color=color[0], alpha=0.25, zorder=2)
    # pp
    axis.plot(cut_bin_cents, v_bin_vals_aco, marker='o', markersize=7, linestyle=':',
              color=color[1], linewidth=2, label=dir + r' pp $aco$', zorder=3)
    axis.fill_between(cut_bin_cents, v_bin_vals_aco - v_bin_errs_aco, v_bin_vals_aco + v_bin_errs_aco,
                      color=color[1], alpha=0.25, zorder=2)

    axis.set_ylabel('Average Signed Acoplanarity (rad)', fontsize=10)


axis.set_xlabel(r'Acoplanarity (rad)', fontsize=10)
axis.legend(fontsize=10, loc='best')
axis.grid(True, alpha=0.3)
axis.set_title(f"R={R}, phi_fence={phi_fence_jet_axis}, Fence_y={fence_y}, particle_rap_min={rap_min_jet_finder}, particle_rap_max={rap_max_jet_finder},\njet_rap_min={rap_min_jet_axis}, jet_rap_max={rap_max_jet_axis}, jetpTmin={jet_minpt}, jetpTmax={jet_maxpt}", fontsize=10)
fig.savefig(f"acoplanarity_{dir}.png", dpi=150, bbox_inches='tight')

fig2 = plt.figure(figsize=(8, 6))
axis2 = fig2.add_subplot()
axis2.hist(v_jet_ys, bins=100, color="g", alpha=0.5, label="pp")
axis2.hist(m_jet_ys, bins=100, color="r", alpha=0.5, label="AA")
axis2.set_xlabel(r"$y_r$")
axis2.legend()
fig2.savefig("rap_hist.png", dpi=150, bbox_inches="tight", pad_inches=0.05)

fig3 = plt.figure(figsize=(8, 6))
axis3 = fig3.add_subplot()
axis3.hist(v_jet_phis, bins=100, color="g", alpha=0.5, label="pp")
axis3.hist(m_jet_phis, bins=100, color="r", alpha=0.5, label="AA")
axis3.set_xlabel(r"$\phi$")
axis3.legend()
fig3.savefig("phi_hist.png", dpi=150, bbox_inches="tight", pad_inches=0.05)

fig4 = plt.figure(figsize=(8, 6))
axis4 = fig4.add_subplot()
axis4.hist(v_jet_pts, bins=100, color="g", alpha=0.5, label="pp")
axis4.hist(m_jet_pts, bins=100, color="r", alpha=0.5, label="AA")
axis4.set_xlabel(r"jet $p_T$")
axis4.legend()
fig4.savefig("pT_hist.png", dpi=150, bbox_inches="tight", pad_inches=0.05)

