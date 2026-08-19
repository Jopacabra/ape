import pyhepmc
import os
import sys
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import fastjet
sys.path.insert(0, '..')
import observables

# ---------------------------------------------------------------------------
# OS-level stdout/stderr suppression (Used to suppress hepmc file C++ output)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_fd_output():
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

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
hepmc_dir = "../results/hepmc/"
label = "30-40%"

# Jet-finder parameters
R = 1.0
pTmin_jet_finder = 0.0
rap_min_jet_finder = 0.0
rap_max_jet_finder = 1.5

# Jet selection cuts
jet_minpt = 10.0
jet_maxpt = 120.0
rap_min_jet_axis = 0.0
rap_max_jet_axis = 1.5
phi_fence_jet_axis = 0.0

# 2-D histogram binning  (relative to jet axis)
n_dphi_bins = 7          # bins in Δφ
n_deta_bins = 7          # bins in Δη
dphi_max    = R       # Δφ ∈ (−π, π]
deta_max    = R        # Δη ∈ (−deta_max, deta_max)

dphi_edges = np.linspace(-dphi_max, dphi_max, n_dphi_bins + 1)
deta_edges = np.linspace(-deta_max, deta_max, n_deta_bins + 1)
dphi_centers = 0.5 * (dphi_edges[:-1] + dphi_edges[1:])
deta_centers = 0.5 * (deta_edges[:-1] + deta_edges[1:])

plot3d = False
gammatagged = True
# ---------------------------------------------------------------------------
# Accumulate histograms
# ---------------------------------------------------------------------------
# Each entry is a 2-D array (n_deta_bins × n_dphi_bins); we keep a running
# weighted sum of particle counts and a running sum of weights so we can
# compute the weighted average at the end.

results = {}   # keyed by case ("v" or "m")

for case in ["v", "m"]:

    hepmc_dir_case   = hepmc_dir + case + "/"
    hepmc_files = os.listdir(hepmc_dir_case)

    # Accumulators
    hist_sum         = np.zeros((n_deta_bins, n_dphi_bins))   # weighted particle counts
    weight_sum       = 0.0                                    # sum of event weights
    n_jets           = 0
    num_constituents = []

    # Counters for diagnostics
    failed = no_pair = n_low_pt = n_high_pt = n_low_rap = n_high_rap = n_low_phi = 0

    for file in hepmc_files:
        try:
            # with suppress_fd_output():
            with pyhepmc.open(hepmc_dir_case + file) as f:
                event = f.read()
        except IsADirectoryError:
            continue

        # ── Jet finding ──────────────────────────────────────────────────────
        try:
            if gammatagged:
                gamma, jet = observables.hepmc_to_fastjet_gamma_jet_pairs(
                    hepmc_event=event, R=R,
                    rap_min=rap_min_jet_finder,
                    rap_max=rap_max_jet_finder,
                    pTmin=pTmin_jet_finder,
                    scheme=fastjet.WTA_pt_scheme,
                )
                jets = [jet]
            else:
                jets = observables.hepmc_to_fastjet(
                    hepmc_event=event, R=R,
                    rap_min=rap_min_jet_finder,
                    rap_max=rap_max_jet_finder,
                    pTmin=pTmin_jet_finder,
                    scheme=fastjet.WTA_pt_scheme,
                )
        except Exception:
            failed += 1
            continue
        for jet in jets:
            if jet is None:
                no_pair += 1
                continue

            weight  = event.weight("pythia")
            jet_pt  = jet.pt()

            # ── Jet pT cut ───────────────────────────────────────────────────────
            if jet_pt < jet_minpt:
                n_low_pt += 1
                continue
            if jet_pt > jet_maxpt:
                n_high_pt += 1
                continue

            # ── Jet φ fence cut ──────────────────────────────────────────────────
            jet_p   = np.array([jet.px(), jet.py(), jet.pz()])
            jet_phi = np.arctan2(jet_p[1], jet_p[0])
            if (np.mod(jet_phi, np.pi / 2) < phi_fence_jet_axis / 2
                    or np.mod(jet_phi, np.pi / 2) > (np.pi / 2 - phi_fence_jet_axis / 2)):
                n_low_phi += 1
                continue

            # ── Jet rapidity cut ─────────────────────────────────────────────────
            jet_e   = jet.e()
            jet_rap = 0.5 * np.log((jet_e + jet_p[2]) / (jet_e - jet_p[2]))
            if abs(jet_rap) < rap_min_jet_axis:
                n_low_rap += 1
                continue
            if abs(jet_rap) > rap_max_jet_axis:
                n_high_rap += 1
                continue

            # ── Build Δφ, Δη for every constituent ───────────────────────────────
            constituents = jet.constituents()
            num_constituents.append(len(constituents))
            if not constituents:
                continue

            dphi_vals = []
            deta_vals = []
            for p in constituents:
                p_phi = np.arctan2(p.py(), p.px())
                p_e   = p.e()
                p_pz  = p.pz()
                # Guard against 0 division
                if abs(p_e - abs(p_pz)) < 1e-10:
                    continue
                p_rap = 0.5 * np.log((p_e + p_pz) / (p_e - p_pz))

                raw_dphi = p_phi - jet_phi
                # Wrap Δφ into (−π, π]
                raw_dphi = (raw_dphi + np.pi) % (2 * np.pi) - np.pi

                dphi_vals.append(raw_dphi)
                deta_vals.append(p_rap - jet_rap)

            if not dphi_vals:
                continue

            dphi_vals = np.array(dphi_vals)
            deta_vals = np.array(deta_vals)

            # Histogram constituents for this jet (unweighted counts per bin)
            h, _, _ = np.histogram2d(
                deta_vals, dphi_vals,
                bins=[deta_edges, dphi_edges],
            )

            # Accumulate weighted sum (weight by the event's cross-section weight)
            hist_sum   += weight * h
            weight_sum += weight
            n_jets     += 1

    # ── Weighted average: particles per jet per bin ───────────────────────────
    # Divide by weight_sum to get the weighted mean particle count per bin,
    # then divide by n_jets to get the average per-jet yield.
    # Equivalently: <dN/(dΔφ dΔη)> = hist_sum / weight_sum  (per-event average)
    # We keep units as "particles per bin" (i.e. not divided by bin area yet).
    if weight_sum > 0:
        avg_hist = hist_sum / weight_sum
    else:
        avg_hist = hist_sum

    results[case] = avg_hist

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print(f"\n── Case: {case} ──────────────────────────────────────")
    print(f"  Files read         : {len(hepmc_files)}")
    print(f"  Failed jet-finding : {failed}")
    if gammatagged:
        print(f"  No γ–jet pair      : {no_pair}")
    print(f"  pT too low/high    : {n_low_pt} / {n_high_pt}")
    print(f"  rap too low/high   : {n_low_rap} / {n_high_rap}")
    print(f"  φ fence rejected   : {n_low_phi}")
    print(f"  Accepted jets      : {n_jets}")
    print(f"  Total weight       : {weight_sum:.4g}")
    print(f"  Average num. const.: {np.mean(num_constituents):.2f}")
    print(f"  Min num. const.     : {np.min(num_constituents):.2f}")
    print(f"  Max num. const.     : {np.max(num_constituents):.2f}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
dphi_cents = 0.5 * (dphi_edges[:-1] + dphi_edges[1:])
deta_cents = 0.5 * (deta_edges[:-1] + deta_edges[1:])

# Shared colour-scale limits (use the medium histogram to set the scale)
vmax = results["m"].max()
vmin = 0.0

if plot3d:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True,
                             subplot_kw={"projection": "3d"})
else:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

panel_data = [
    (results["v"],                "pp (vacuum)",          "Blues"),
    (results["m"],                "AA (medium)",          "Reds"),
    (results["m"] - results["v"], "AA − pp (difference)", "RdBu_r"),
]

for ax, (data, title, cmap) in zip(axes, panel_data):
    if plot3d:
        if "difference" in title:
            # Symmetric colour scale for the difference panel
            absmax = np.abs(data).max()
            im = ax.plot_surface(
                dphi_centers, deta_centers, data,
                cmap=cmap, vmin=-absmax, vmax=absmax,
            )
        else:
            im = ax.plot_surface(
                dphi_centers, deta_centers, data,
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
        ax.set_zscale("log")
        # theta = np.linspace(0, 2 * np.pi, 300)
        # ax.plot(R * np.cos(theta), R * np.sin(theta), 0,
        #         "k--", linewidth=1, alpha=0.6, label=f"R={R}")
    else:
        if "difference" in title:
            # Symmetric colour scale for the difference panel
            absmax = np.abs(data).max()
            im = ax.pcolormesh(
                dphi_edges, deta_edges, data,
                cmap=cmap, vmin=-absmax, vmax=absmax, shading="flat",
            )
        else:
            im = ax.pcolormesh(
                dphi_edges, deta_edges, data,
                cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
            )

        fig.colorbar(im, ax=ax, label="Avg. particles / bin")
        # Mark the jet-cone boundary
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R * np.cos(theta), R * np.sin(theta),
                "k--", linewidth=1, alpha=0.6, label=f"R={R}")


    ax.set_xlabel(r"$\Delta\phi$", fontsize=13)
    ax.set_ylabel(r"$\Delta\eta$", fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-dphi_max, dphi_max)
    ax.set_ylim(-deta_max, deta_max)


fig.suptitle(
    f"{label} Cent., Avg. particle density around jet axis\n"
    f"R={R}, pT∈[{jet_minpt},{jet_maxpt}] GeV, "
    f"|y_jet|∈[{rap_min_jet_axis},{rap_max_jet_axis}], "
    f"φ-fence={phi_fence_jet_axis}",
    fontsize=11,
)

out_file = "intrajet_dN_dphi_deta_2d.png"
fig.savefig(out_file, dpi=150, bbox_inches="tight")
print(f"\nSaved figure → {out_file}")
plt.show()