import pyhepmc
import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import io
import matplotlib.pyplot as plt
sys.path.insert(0, '..')
import observables

# ─── Settings ─────────────────────────────────────────────────────────────────

# Which flow harmonic to compute (n=2 → elliptic flow v2, n=3 → v3, …)
n_harmonic = 2
print(f"n_harmonic = {n_harmonic}")

# results directory for HepMC files
hepmc_dir = f"../results_saved/30_40_avg_AuAu_post_finkin_fix/hepmc/"
# hepmc_dir = "../results/hepmc/m/"
cent_label = "30-40%"

# Bootstrap statistics
n_bootstrap_samples = 1000
batch_size = 1  # Number of hard events to combine for bootstrapping

# Particle cuts (hard sector)
pids = [211]           # charged pions; extend as needed
pTmin = 3.0
pTmax = 12.0
rapmin = 0.0
rapmax = 1000
num_pt_bins = 3
pT_bins = np.linspace(pTmin, pTmax, num_pt_bins + 1)
pT_bin_cents = 0.5 * (pT_bins[:-1] + pT_bins[1:])

# ─── Bootstrapper ────────────────────────────────────────────────────────

def bootstrap_vn(Qx, Qy, M, n_resamples, rng):
    """
    Bootstrap the event-plane vn estimator.
    Per event: vn ≈ sqrt(Qx² + Qy²) / M  (simplified scalar event-plane).
    Resamples events, computes weighted-sum estimator per resample.
    """
    n_events = Qx.shape[0]
    dist = np.empty((n_resamples, Qx.shape[1]), dtype=np.float64)
    print(n_events)
    for i in range(n_resamples):
        # Select random events, with replacement, for bootstrap samples
        idx = rng.integers(0, n_events, size=n_events)
        sQx = Qx[idx].sum(axis=0)
        sQy = Qy[idx].sum(axis=0)
        sM = M[idx].sum(axis=0)

        # Compute the observable for this bootstrap sample
        with np.errstate(invalid="ignore", divide="ignore"):
            dist[i] = np.where(sM > 0, np.sqrt(sQx ** 2 + sQy ** 2) / sM, np.nan)

    # Take the mean and std deviation of the samples
    vn_mean = np.nanmean(dist, axis=0)
    vn_std = np.nanstd(dist, axis=0)
    return vn_mean, vn_std

# ─── HEPData CSV parser ────────────────────────────────────────────────────────

def parse_hepdata_csv(filepath):
    """Parse a multi-block HEPData CSV file into a dict of DataFrames keyed by
    the centrality label found in '#: Centrality' comment lines.
    Blocks whose label is empty (the pT-axis-only header block) are skipped."""
    datasets = {}
    current_centrality = None
    current_lines = []
    header = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#: Centrality"):
                # Save previous block if it had a non-empty label and data
                if current_centrality and current_lines and header:
                    block_text = "\n".join([header] + current_lines)
                    df = pd.read_csv(io.StringIO(block_text))
                    datasets[current_centrality] = df
                # Extract centrality label (last comma-separated field)
                label = line.split(",")[-1].strip()
                # Empty label → this is the axis-definition block; skip it
                current_centrality = label if label else None
                current_lines = []
                header = None
                continue

            if line.startswith("#"):
                continue

            if not line.strip():
                continue

            if current_centrality is not None:
                if header is None:
                    header = line
                else:
                    current_lines.append(line)

    # Save last block
    if current_centrality and current_lines and header:
        block_text = "\n".join([header] + current_lines)
        df = pd.read_csv(io.StringIO(block_text))
        datasets[current_centrality] = df

    return datasets

# ─── Plotting ─────────────────────────────────────────────────────────────────

fig, axis = plt.subplots(figsize=(8, 6))
plot_colors = ["blue", "red"]

# ── Experimental data from HEPData ────────────────────────────────────────────
filepath = "HEPData-ins850211-v1-csv/Figure3a.csv"
datasets = parse_hepdata_csv(filepath)

if datasets and n_harmonic == 2:
    centralities = ["5-10", "30-35", "50-60"]
    colors = cm.plasma(np.linspace(0.05, 0.90, len(datasets)))
    for (centrality, df), color in zip(datasets.items(), colors):
        if centrality not in centralities:
            continue
        pt       = df["$p_T$ (GeV/$c$)"].values
        vn       = df["$v_2$"].values
        stat_up  = df["stat. +"].values
        stat_dn  = np.abs(df["stat. -"].values)

        axis.errorbar(
            pt, vn,
            yerr=[stat_dn, stat_up],
            fmt="o",
            color=color,
            markersize=4,
            linewidth=1.2,
            capsize=2,
            label=f"PHENIX {centrality}%",
        )
        axis.set_xlim(left=0, right=pTmax+1)
elif n_harmonic != 2:
    print(f"No exp. data for n={n_harmonic}")
    axis.set_xlim(left=pTmin, right=pTmax)
else:
    print(f"Warning: no data blocks found in {filepath}")

# ─── Loop over events ────────────────────────────────────────────────
for j, case in enumerate(["v", "m"]):
    case_dir = hepmc_dir + case + "/"
    hepmc_files = os.listdir(case_dir)

    Qx_batches = []
    Qy_batches = []
    M_batches  = []

    event_Qx = []
    event_Qy = []
    event_M  = []

    event_count = 0
    n_failed    = 0
    max_files   = 15000

    num_files = len(hepmc_files)
    print(f"Number of AuAu files: {num_files}")
    if max_files < num_files:
        print(f"Limiting to {max_files} files!")

    for file in hepmc_files[: min(max_files, num_files)]:
        try:
            with pyhepmc.open(case_dir + file) as f:
                event = f.read()

            try:
                Qx, Qy, M, weight = observables.compute_event_vn(
                    hepmc_event=event,
                    n=n_harmonic,
                    rap_min=rapmin, rap_max=rapmax,
                    pTmin=pTmin, pTmax=pTmax,
                    pT_bins=pT_bins,
                    include=pids,
                )
            except Exception:
                n_failed += 1
                del event
                continue

            # Collect q vectors and particle numbers, weighted by pythia event weight
            event_Qx.append((Qx * weight).astype(np.float32))
            event_Qy.append((Qy * weight).astype(np.float32))
            event_M.append((M * weight).astype(np.float32))
            event_count += 1

            # Combine events into batches for bootstrapping -- usually one event per batch
            if event_count % batch_size == 0:
                Qx_batches.append(np.array(event_Qx, dtype=np.float32))
                Qy_batches.append(np.array(event_Qy, dtype=np.float32))
                M_batches.append(np.array(event_M, dtype=np.float32))
                event_Qx, event_Qy, event_M = [], [], []

            del event

        except IsADirectoryError:
            continue

    # Flush remaining events, even if they didn't make a full batch
    if event_Qx:
        Qx_batches.append(np.array(event_Qx, dtype=np.float32))
        Qy_batches.append(np.array(event_Qy, dtype=np.float32))
        M_batches.append(np.array(event_M, dtype=np.float32))

    print(f"{n_failed} failed AuAu events out of {event_count + n_failed}")

    # ─── Flatten ──────────────────────────────────────────────────────────────────

    all_Qx = np.concatenate(Qx_batches, axis=0)  # (N_events, n_bins)
    all_Qy = np.concatenate(Qy_batches, axis=0)
    all_M  = np.concatenate(M_batches, axis=0)
    del Qx_batches, Qy_batches, M_batches
    gc.collect()

    # ─── Bootstrap ────────────────────────────────────────────────────────────────

    rng = np.random.default_rng()

    print(f"Bootstrapping v{n_harmonic}…")
    vn_mean, vn_std = bootstrap_vn(all_Qx, all_Qy, all_M, n_bootstrap_samples, rng)
    del all_Qx, all_Qy, all_M
    gc.collect()

    x = np.repeat(pT_bins, 2)[1:-1]  # shape (2N,)
    y_mean = np.repeat(vn_mean, 2)
    y_upper = np.repeat(vn_mean + vn_std, 2)
    y_lower = np.repeat(vn_mean - vn_std, 2)
    axis.fill_between(x, y_lower, y_upper, color=plot_colors[j], alpha=0.3,
                      linewidth=0)
    axis.plot(x, y_mean, color=plot_colors[j], lw=1.8, label=rf"APE {case} $v_{{{n_harmonic}}}$ ({cent_label})")

# ── Plot APE result ──────────────────────────────────────────────────────
# axis.errorbar(
#     pT_bins, vn_mean, yerr=vn_std,
#     marker="o", linestyle="-", color="red", linewidth=1.5,
#     label=f"APE $v_{{{n_harmonic}}}$ (0–10%)",
# )


axis.set_xlabel(r"$p_T$ (GeV/$c$)", fontsize=14)
axis.set_ylabel(rf"$v_{{{n_harmonic}}}$", fontsize=14)
axis.set_ylim(bottom=0)
axis.grid(True, linestyle=":", alpha=0.5)
axis.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.85)
axis.set_title(rf"Flow harmonic $v_{{{n_harmonic}}}$ of hard particles (AuAu)", fontsize=13)

fig.tight_layout()
fig.savefig(f"vn{n_harmonic}.png", dpi=150)
print(f"Saved vn{n_harmonic}.png")