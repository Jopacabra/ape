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

# results subdirectory for HepMC files
name = "hepmc"

# Bootstrap statistics
n_bootstrap_samples = 1000
batch_size = 1000

# Particle cuts (hard sector)
pids = [211]           # charged pions; extend as needed
pTmin = 3.0
pTmax = 12.0
rapmin = 0.0
rapmax = 1000
num_pt_bins = 3
pT_bins = np.linspace(pTmin, pTmax, num_pt_bins + 1)
pT_bin_cents = 0.5 * (pT_bins[:-1] + pT_bins[1:])

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

# ─── Per-event vn accumulator ─────────────────────────────────────────────────

def compute_event_vn(hepmc_event, n, rap_min, rap_max, pTmin, pTmax, pT_bins, include):
    """
    Compute the per-pT-bin Q-vector components for harmonic n from a single
    HepMC event, weighted by the Pythia event weight.

    Returns
    -------
    Qx : ndarray, shape (n_bins,)   sum of cos(n*phi) for particles in each bin
    Qy : ndarray, shape (n_bins,)   sum of sin(n*phi) for particles in each bin
    M  : ndarray, shape (n_bins,)   particle multiplicity in each bin
    weight : float                  Pythia event weight
    """
    n_bins = len(pT_bins) - 1
    Qx = np.zeros(n_bins, dtype=np.float64)
    Qy = np.zeros(n_bins, dtype=np.float64)
    M  = np.zeros(n_bins, dtype=np.float64)

    weight = hepmc_event.weight("pythia")

    for particle in hepmc_event.particles:
        if particle.status != 1:          # final-state only
            continue
        pid = particle.pid
        if include and pid not in include:
            continue

        mom = particle.momentum
        pT  = np.sqrt(mom.px**2 + mom.py**2)
        if pT < pTmin or pT >= pTmax:
            continue

        # rapidity
        try:
            rap = 0.5 * np.log((mom.e + mom.pz) / (mom.e - mom.pz))
        except (ZeroDivisionError, ValueError):
            continue
        if abs(rap) < rap_min or abs(rap) > rap_max:
            continue

        phi  = np.arctan2(mom.py, mom.px)
        ibin = np.searchsorted(pT_bins[1:], pT)   # bin index
        ibin = min(ibin, n_bins - 1)

        Qx[ibin] += np.cos(n * phi)
        Qy[ibin] += np.sin(n * phi)
        M[ibin]  += 1.0

    return Qx, Qy, M, weight

# ─── Loop over AuAu (m) events ────────────────────────────────────────────────

hepmc_dir = f"../results/{name}/m/"
hepmc_files = os.listdir(hepmc_dir)

event_Qx_batches = []
event_Qy_batches = []
event_M_batches  = []

batch_Qx = []
batch_Qy = []
batch_M  = []

event_count = 0
n_failed    = 0
max_files   = 10000

num_files = len(hepmc_files)
print(f"Number of AuAu files: {num_files}")
if max_files < num_files:
    print(f"Limiting to {max_files} files!")

for file in hepmc_files[: min(max_files, num_files)]:
    try:
        with pyhepmc.open(hepmc_dir + file) as f:
            event = f.read()

        try:
            Qx, Qy, M, weight = compute_event_vn(
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

        batch_Qx.append((Qx * weight).astype(np.float32))
        batch_Qy.append((Qy * weight).astype(np.float32))
        batch_M.append((M  * weight).astype(np.float32))
        event_count += 1

        if event_count % batch_size == 0:
            event_Qx_batches.append(np.array(batch_Qx, dtype=np.float32))
            event_Qy_batches.append(np.array(batch_Qy, dtype=np.float32))
            event_M_batches.append( np.array(batch_M,  dtype=np.float32))
            batch_Qx, batch_Qy, batch_M = [], [], []

        del event

    except IsADirectoryError:
        continue

# Flush remaining
if batch_Qx:
    event_Qx_batches.append(np.array(batch_Qx, dtype=np.float32))
    event_Qy_batches.append(np.array(batch_Qy, dtype=np.float32))
    event_M_batches.append( np.array(batch_M,  dtype=np.float32))

print(f"{n_failed} failed AuAu events out of {event_count + n_failed}")

# ─── Flatten ──────────────────────────────────────────────────────────────────

all_Qx = np.concatenate(event_Qx_batches, axis=0)  # (N_events, n_bins)
all_Qy = np.concatenate(event_Qy_batches, axis=0)
all_M  = np.concatenate(event_M_batches,  axis=0)
del event_Qx_batches, event_Qy_batches, event_M_batches
gc.collect()

# ─── Bootstrap ────────────────────────────────────────────────────────────────

rng = np.random.default_rng()

def bootstrap_vn(Qx, Qy, M, n_resamples, rng):
    """
    Bootstrap the event-plane vn estimator.
    Per event: vn ≈ sqrt(Qx² + Qy²) / M  (simplified scalar event-plane).
    Resamples events, computes weighted-sum estimator per resample.
    """
    n_events = Qx.shape[0]
    dist = np.empty((n_resamples, Qx.shape[1]), dtype=np.float64)

    for i in range(n_resamples):
        idx = rng.integers(0, n_events, size=n_events)
        sQx = Qx[idx].sum(axis=0)
        sQy = Qy[idx].sum(axis=0)
        sM  = M[idx].sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            dist[i] = np.where(sM > 0, np.sqrt(sQx**2 + sQy**2) / sM, np.nan)

    vn_mean = np.nanmean(dist, axis=0)
    vn_std  = np.nanstd(dist,  axis=0)
    return vn_mean, vn_std


print(f"Bootstrapping v{n_harmonic}…")
vn_mean, vn_std = bootstrap_vn(all_Qx, all_Qy, all_M, n_bootstrap_samples, rng)
del all_Qx, all_Qy, all_M
gc.collect()

# ─── Plotting ─────────────────────────────────────────────────────────────────

fig, axis = plt.subplots(figsize=(8, 6))

# ── Experimental data from HEPData ────────────────────────────────────────────
filepath = "HEPData-ins850211-v1-csv/Figure3a.csv"
datasets = parse_hepdata_csv(filepath)

if datasets:
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
else:
    print(f"Warning: no data blocks found in {filepath}")

# ── APE simulation result ──────────────────────────────────────────────────────
axis.errorbar(
    pT_bin_cents, vn_mean, yerr=vn_std,
    marker="o", linestyle="-", color="red", linewidth=1.5,
    label=f"APE $v_{{{n_harmonic}}}$ (0–10%)",
)

axis.set_xlabel(r"$p_T$ (GeV/$c$)", fontsize=14)
axis.set_ylabel(rf"$v_{{{n_harmonic}}}$", fontsize=14)
axis.set_ylim(bottom=0)
axis.grid(True, linestyle=":", alpha=0.5)
axis.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.85)
axis.set_title(rf"Flow harmonic $v_{{{n_harmonic}}}$ of hard particles (AuAu)", fontsize=13)

fig.tight_layout()
fig.savefig(f"vn{n_harmonic}.png", dpi=150)
print(f"Saved vn{n_harmonic}.png")