import pyhepmc
import os
import sys
import contextlib
import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import fastjet
sys.path.insert(0, '..')
import observables


@contextlib.contextmanager
def suppress_fd_output():
    """Suppresses stdout and stderr at the OS file-descriptor level.
    Works even for output from compiled C/C++ extensions."""
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    saved_fds = [os.dup(1), os.dup(2)]
    try:
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        for fd in null_fds + saved_fds:
            os.close(fd)


def weighted_mean(acos, weights):
    """Weighted mean of acos, used as the bootstrap statistic."""
    return np.nansum(acos * weights) / np.nansum(weights)


def compute_bin_vals(acos, weights, bin_edges, bin_in_jet_x, analyzed_x=None, delta_phi=1):
    """Compute the array of per-bin values (weighted sum or weighted mean)."""
    vals = []
    for j, bin_lo in enumerate(bin_edges[:-1]):
        bin_hi = bin_edges[j + 1]
        if bin_in_jet_x:
            cut = (analyzed_x > bin_lo) & (analyzed_x <= bin_hi)
            w = weights[cut]
            a = acos[cut]
            val = weighted_mean(a, w) if len(w) >= 2 else np.nan
        else:
            cut = (acos > bin_lo) & (acos <= bin_hi)
            w = weights[cut]
            val = np.nansum(w) / (np.nansum(weights) * delta_phi)
        vals.append(val)
    return np.array(vals)


def bootstrap_ratio_or_diff(
    m_acos, m_weights, m_x,
    v_acos, v_weights, v_x,
    bin_edges, bin_in_jet_x,
    n_resamples, mode, rng,
):
    """
    Bootstrap the error of the per-bin ratio (m/v) or difference (m-v).

    Resamples each dataset independently (they are independent), recomputes
    the derived quantity for each resample, and returns the standard deviation
    across resamples as the error for each bin.

    Parameters
    ----------
    mode : str
        "ratio" or "diff"

    Returns
    -------
    central : np.ndarray  — central-value array (shape: n_bins)
    errs    : np.ndarray  — bootstrap standard-error array (shape: n_bins)
    """
    n_m = len(m_acos)
    n_v = len(v_acos)

    central_m = compute_bin_vals(m_acos, m_weights, bin_edges, bin_in_jet_x, m_x, delta_phi=delta_phi)
    central_v = compute_bin_vals(v_acos, v_weights, bin_edges, bin_in_jet_x, v_x, delta_phi=delta_phi)

    if mode == "ratio":
        central = central_m / central_v
    else:
        central = central_m - central_v

    boot_stats = np.empty((n_resamples, len(bin_edges) - 1))
    for i in range(n_resamples):
        idx_m = rng.integers(0, n_m, size=n_m)
        idx_v = rng.integers(0, n_v, size=n_v)

        bm_acos    = m_acos[idx_m]
        bm_weights = m_weights[idx_m]
        bm_x       = m_x[idx_m] if m_x is not None else None

        bv_acos    = v_acos[idx_v]
        bv_weights = v_weights[idx_v]
        bv_x       = v_x[idx_v] if v_x is not None else None

        bm_vals = compute_bin_vals(bm_acos, bm_weights, bin_edges, bin_in_jet_x, bm_x, delta_phi=delta_phi)
        bv_vals = compute_bin_vals(bv_acos, bv_weights, bin_edges, bin_in_jet_x, bv_x, delta_phi=delta_phi)

        if mode == "ratio":
            boot_stats[i] = bm_vals / bv_vals
        else:
            boot_stats[i] = bm_vals - bv_vals

    errs = np.nanstd(boot_stats, axis=0)
    return central, errs


# --- Configuration ---

# Results subdirectory for HepMC files
hepmc_dir = "../results_saved/30_40_avg_AuAu_post_finkin_fix/hepmc/"
label = "30-40%"

# Statistics settings
n_bootstrap_samples = 10000

# Jet finding cuts
R = 1.0
pTmin_jet_finder = 2.0      # Minimum pT of constituents to consider
rap_min_jet_finder = 0.0    # Minimum rapidity of constituents to consider
rap_max_jet_finder = 1.5    # Maximum rapidity of constituents to consider

# Jet axis cuts
jet_minpt = 10.0
jet_maxpt = 120.0
rap_min_jet_axis = 0.0
rap_max_jet_axis = 1.5
phi_fence_jet_axis = 0.4
fence_y = True

# Acoplanarity (delta-j) binning
aco_min = -0.5
aco_max = +0.5
num_aco_bins = 30
aco_bins = np.linspace(aco_min, aco_max, num_aco_bins)
delta_phi = aco_bins[1] - aco_bins[0]

# jet_x binning:
#   False  -> bin by aco, plot weighted sum of events per bin (original behaviour)
#   True   -> bin by jet_x, plot weighted mean aco per bin with bootstrapped errors
bin_in_jet_x = False
jet_x_min = 0.0
jet_x_max = 1.0
num_jet_x_bins = 10
jet_x_bins = np.linspace(jet_x_min, jet_x_max, num_jet_x_bins)

# Observables settings
deflection_dir = "phi"  # Direction for signed deflection computation
pt_weighting = 0        # Weight each particle's phase by p_T^(pt_weighting)
plot_group = "both"    # "ratio", "diff", "both", "m", or "v"

# Plot x-axis range (used only when bin_in_jet_x = False)
xmin = -0.5
xmax = +0.5

# Plotting style
flow_i = 0
colors = ["r", "g", "b", "m"]

# --- Storage arrays ---

v_jet_pts, v_jet_ys, v_jet_phis, v_jet_xs = [], [], [], []
m_jet_pts, m_jet_ys, m_jet_phis, m_jet_xs = [], [], [], []

# Per-case raw data stored for possible bootstrap use later
case_data = {}

bin_vals_aco,   bin_errs_aco   = [], []
v_bin_vals_aco, v_bin_errs_aco = [], []

# --- Main loop ---

for case in ["v", "m"]:

    case_dir = hepmc_dir + case + "/"
    hepmc_files = os.listdir(case_dir)

    # Per-case counters
    n_failed        = 0
    n_no_pair       = 0
    n_subjets       = 0
    n_too_small_pt  = 0
    n_too_large_pt  = 0
    n_too_small_rap = 0
    n_too_large_rap = 0
    n_too_small_phi = 0
    n_too_large_phi = 0

    # Per-case accepted jet lists
    analyzed_WTA_jets    = []
    analyzed_ES_jets     = []
    analyzed_WTA_jet_pts = []
    analyzed_ES_jet_pts  = []
    analyzed_x           = []
    analyzed_weights     = []

    for file in hepmc_files:
        try:
            with suppress_fd_output():
                with pyhepmc.open(case_dir + file) as f:
                    event = f.read()

            # --- Jet finding ---
            try:
                gamma, WTA_jet = observables.hepmc_to_fastjet_gamma_jet_pairs(
                    hepmc_event=event,
                    R=R,
                    rap_min=rap_min_jet_finder,
                    rap_max=rap_max_jet_finder,
                    pTmin=pTmin_jet_finder,
                    scheme=fastjet.WTA_pt_scheme,
                )

                # Recluster with E-scheme to get axis
                escheme_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R, fastjet.E_scheme)
                try:
                    new_jets = escheme_def(WTA_jet.constituents())
                    if len(new_jets) > 1:
                        n_subjets += 1
                        continue
                    ES_jet = new_jets[0]
                except AttributeError:
                    # WTA_jet has no constituents — it is a single-particle jet
                    ES_jet = WTA_jet

            except Exception:
                n_failed += 1
                continue

            if WTA_jet is None:
                n_no_pair += 1
                continue

            # --- Jet cuts ---
            jet_pt = WTA_jet.pt()
            if jet_pt < jet_minpt:
                n_too_small_pt += 1
                continue
            elif jet_pt > jet_maxpt:
                n_too_large_pt += 1
                continue

            jet_p   = np.array([WTA_jet.px(), WTA_jet.py(), WTA_jet.pz()])
            jet_phi = np.arctan2(jet_p[1], jet_p[0])
            if np.mod(jet_phi, np.pi / 2) < phi_fence_jet_axis / 2:
                n_too_small_phi += 1
                continue
            elif fence_y and np.mod(jet_phi, np.pi / 2) > (np.pi / 2 - phi_fence_jet_axis / 2):
                n_too_large_phi += 1
                continue

            jet_rap = WTA_jet.eta()
            if abs(jet_rap) < rap_min_jet_axis:
                n_too_small_rap += 1
                continue
            elif abs(jet_rap) > rap_max_jet_axis:
                n_too_large_rap += 1
                continue

            # --- Accepted jet ---
            jet_x  = WTA_jet.pt() / gamma.pt()
            weight = event.weight("pythia")

            analyzed_WTA_jets.append(WTA_jet)
            analyzed_ES_jets.append(ES_jet)
            analyzed_WTA_jet_pts.append(jet_pt)
            analyzed_ES_jet_pts.append(ES_jet.pt())
            analyzed_x.append(jet_x)
            analyzed_weights.append(weight)

            if case == "v":
                v_jet_pts.append(jet_pt)
                v_jet_phis.append(jet_phi)
                v_jet_ys.append(jet_rap)
                v_jet_xs.append(jet_x)
            else:
                m_jet_pts.append(jet_pt)
                m_jet_phis.append(jet_phi)
                m_jet_ys.append(jet_rap)
                m_jet_xs.append(jet_x)

        except IsADirectoryError:
            continue

    # --- Readout ---
    print(f"\n=== Case: {case} ===")
    print(f"  Files:                      {len(hepmc_files)}")
    print(f"  Failed events:              {n_failed}")
    print(f"  No gamma-jet pair:          {n_no_pair}")
    print(f"  Split reclustered jets:     {n_subjets}")
    print(f"  Too small pT:               {n_too_small_pt}")
    print(f"  Too large pT:               {n_too_large_pt}")
    print(f"  Too small rapidity:         {n_too_small_rap}")
    print(f"  Too large rapidity:         {n_too_large_rap}")
    print(f"  Too small phi (fence):      {n_too_small_phi}")
    print(f"  Too large phi (fence):      {n_too_large_phi}")
    print(f"  Accepted jets:              {len(analyzed_WTA_jets)}")

    n_constituents = [len(list(j.constituents())) for j in analyzed_WTA_jets]
    print(f"  Mean constituents:          {np.mean(n_constituents):.2f}")
    print(f"  Fraction with N<=2:         {np.mean(np.array(n_constituents) <= 2):.2f}")

    # --- Compute signed deflection for each accepted jet ---
    acos = np.array([
        observables.signed_deflection(jet1=WTA_jet, jet2=ES_jet, dir=deflection_dir)
        for ES_jet, WTA_jet in zip(analyzed_ES_jets, analyzed_WTA_jets)
    ], dtype=np.float32)

    print(f"  aco mean: {np.nanmean(acos):.4f}  max: {np.nanmax(acos):.4f}  min: {np.nanmin(acos):.4f}")

    analyzed_weights     = np.array(analyzed_weights)
    analyzed_WTA_jet_pts = np.array(analyzed_WTA_jet_pts)
    analyzed_ES_jet_pts  = np.array(analyzed_ES_jet_pts)
    analyzed_x           = np.array(analyzed_x)

    # Store raw per-event data for bootstrap use when plot_group is "ratio" or "diff"
    case_data[case] = {
        "acos":    acos,
        "weights": analyzed_weights,
        "x":       analyzed_x,
    }

    # --- Binning (Poisson-style errors; used for "both", "m", "v") ---
    active_bin_edges = jet_x_bins if bin_in_jet_x else aco_bins

    if bin_in_jet_x:
        # Bin by jet_x; compute weighted mean aco per bin with bootstrapped error
        for j, bin_lo in enumerate(active_bin_edges[:-1]):
            bin_hi = active_bin_edges[j + 1]
            cut    = (analyzed_x > bin_lo) & (analyzed_x <= bin_hi)
            w      = analyzed_weights[cut]
            a      = acos[cut]

            if len(w) < 2:
                val, err = np.nan, np.nan
            else:
                val = weighted_mean(a, w)
                rng = np.random.default_rng()
                boot_result = bootstrap(
                    (a, w),
                    weighted_mean,
                    n_resamples=n_bootstrap_samples,
                    paired=True,
                    rng=rng,
                )
                err = boot_result.standard_error

            if case == "v":
                v_bin_vals_aco.append(val)
                v_bin_errs_aco.append(err)
            else:
                bin_vals_aco.append(val)
                bin_errs_aco.append(err)
    else:
        # Bin by aco; compute weighted sum of events per bin (Poisson error)
        for j, bin_lo in enumerate(active_bin_edges[:-1]):
            bin_hi = active_bin_edges[j + 1]
            cut    = (acos > bin_lo) & (acos <= bin_hi)
            w      = analyzed_weights[cut]

            val = np.nansum(w)
            err = np.sqrt(np.nansum(w ** 2))

            if case == "v":
                v_bin_vals_aco.append(val)
                v_bin_errs_aco.append(err)
            else:
                bin_vals_aco.append(val)
                bin_errs_aco.append(err)

# --- Convert to arrays ---
bin_vals_aco   = np.array(bin_vals_aco)
bin_errs_aco   = np.array(bin_errs_aco)
v_bin_vals_aco = np.array(v_bin_vals_aco)
v_bin_errs_aco = np.array(v_bin_errs_aco)

active_bins = jet_x_bins if bin_in_jet_x else aco_bins
bin_centers = (active_bins[1:] + active_bins[:-1]) / 2
x_label     = r'jet $x_j$'            if bin_in_jet_x else r'$\Delta j$ (rad)'
y_label_n   = r'Mean $\Delta j$ (rad)' if bin_in_jet_x else r'(1/N)dN/d$\Delta j$'
plot_xmin   = jet_x_min                if bin_in_jet_x else xmin
plot_xmax   = jet_x_max                if bin_in_jet_x else xmax

# --- Main plot ---
fig, axis = plt.subplots(figsize=(12 / 2, 7 / 2))
axis.axhline(y=0, color='black', linewidth=1.5, linestyle='--', zorder=1, alpha=0.5)

if plot_group in ("ratio", "diff"):
    # Bootstrap the error of the derived quantity directly
    rng = np.random.default_rng()
    derived_vals, derived_errs = bootstrap_ratio_or_diff(
        m_acos    = case_data["m"]["acos"],
        m_weights = case_data["m"]["weights"],
        m_x       = case_data["m"]["x"],
        v_acos    = case_data["v"]["acos"],
        v_weights = case_data["v"]["weights"],
        v_x       = case_data["v"]["x"],
        bin_edges    = active_bins,
        bin_in_jet_x = bin_in_jet_x,
        n_resamples  = n_bootstrap_samples,
        mode         = plot_group,
        rng          = rng,
    )

    if plot_group == "ratio":
        axis.plot(bin_centers, derived_vals, marker='o', markersize=7, linestyle='-',
                  color=colors[flow_i], linewidth=2, label=deflection_dir + r' $\Delta j$', zorder=3)
        axis.fill_between(bin_centers, derived_vals - derived_errs, derived_vals + derived_errs,
                          color=colors[3], alpha=0.25, zorder=2)
        axis.set_ylabel(r'$R_{\Delta j}$', fontsize=10)
    else:
        axis.plot(bin_centers, derived_vals, marker='o', markersize=7, linestyle='-',
                  color=colors[flow_i], linewidth=2, label=deflection_dir + r' $\Delta \Delta j$', zorder=3)
        axis.fill_between(bin_centers, derived_vals - derived_errs, derived_vals + derived_errs,
                          color=colors[flow_i], alpha=0.25, zorder=2)
        axis.set_ylabel(r'Change in Average Signed $\Delta j$ (rad)', fontsize=10)

elif plot_group == "m":
    axis.plot(bin_centers, bin_vals_aco, marker='o', markersize=7, linestyle='-',
              color=colors[flow_i], linewidth=2, label=deflection_dir + r' AA', zorder=3)
    axis.fill_between(bin_centers, bin_vals_aco - bin_errs_aco, bin_vals_aco + bin_errs_aco,
                      color=colors[flow_i], alpha=0.25, zorder=2)
    axis.set_ylabel(y_label_n, fontsize=10)

elif plot_group == "v":
    axis.plot(bin_centers, v_bin_vals_aco, marker='o', markersize=7, linestyle='-',
              color=colors[1], linewidth=2, label=deflection_dir + r' pp', zorder=3)
    axis.fill_between(bin_centers, v_bin_vals_aco - v_bin_errs_aco, v_bin_vals_aco + v_bin_errs_aco,
                      color=colors[1], alpha=0.25, zorder=2)
    axis.set_ylabel(y_label_n, fontsize=10)

elif plot_group == "both":
    axis.plot(bin_centers, bin_vals_aco, marker='o', markersize=7, linestyle='-',
              color=colors[0], linewidth=2, label=deflection_dir + r' AA', zorder=3)
    axis.fill_between(bin_centers, bin_vals_aco - bin_errs_aco, bin_vals_aco + bin_errs_aco,
                      color=colors[0], alpha=0.25, zorder=2)
    axis.plot(bin_centers, v_bin_vals_aco, marker='o', markersize=7, linestyle=':',
              color=colors[1], linewidth=2, label=deflection_dir + r' pp', zorder=3)
    axis.fill_between(bin_centers, v_bin_vals_aco - v_bin_errs_aco, v_bin_vals_aco + v_bin_errs_aco,
                      color=colors[1], alpha=0.25, zorder=2)
    axis.set_ylabel(y_label_n, fontsize=10)

axis.set_xlabel(x_label, fontsize=10)
axis.legend(fontsize=10, loc='best')
axis.grid(True, alpha=0.3)
axis.set_xlim(plot_xmin, plot_xmax)
axis.set_title(
    f"{label} Cent., R={R}, phi_fence={phi_fence_jet_axis}, Fence_y={fence_y}, "
    f"particle_rap=[{rap_min_jet_finder}, {rap_max_jet_finder}],\n"
    f"jet_rap=[{rap_min_jet_axis}, {rap_max_jet_axis}], "
    f"jet_pT=[{jet_minpt}, {jet_maxpt}]",
    fontsize=10,
)

bin_suffix = "xj" if bin_in_jet_x else deflection_dir
fig.savefig(f"delta_j_{bin_suffix}.png", dpi=150, bbox_inches='tight')

# --- Diagnostic histograms ---
for v_data, m_data, xlabel, fname in [
    (v_jet_ys,   m_jet_ys,   r"$y$",       "hist_rap.png"),
    (v_jet_phis, m_jet_phis, r"$\phi$",    "hist_phi.png"),
    (v_jet_pts,  m_jet_pts,  r"jet $p_T$", "hist_pT.png"),
    (v_jet_xs,   m_jet_xs,   r"jet $x_j$", "hist_xj.png"),
]:
    bins = np.arange(0, 1, 0.05) if "x_j" in xlabel else 100
    fig_h, ax_h = plt.subplots(figsize=(8, 6))
    ax_h.hist(v_data, bins=bins, color="g", alpha=0.5, label="pp")
    ax_h.hist(m_data, bins=bins, color="r", alpha=0.5, label="AA")
    ax_h.set_xlabel(xlabel)
    ax_h.legend()
    fig_h.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0.05)