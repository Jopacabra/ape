import pyhepmc
import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import io
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
import observables

# Plot data?
plot_data = True

# results subdirectory for HepMC files
hepmc_dir = "../results/hepmc/"  #"../results_saved/30_40_avg_AuAu_ogatta_1_200GeV_gammajets/hepmc/"
# hepmc_dir = f"../results/hepmc/"
label = "APE 30-40%"
max_files = 20000

# Statistics settings
n_bootstrap_samples = 1000

# Particle cuts
pids = [211]
pTmin_RAA = 1.0  # Anything below here is liable to be junk both theoretically and computationally.
pTmax_RAA = 16.0
rapmin_RAA = 0.0
rapmax_RAA = 1000
num_pt_bins = 3
pT_bins = np.linspace(pTmin_RAA, pTmax_RAA, num_pt_bins+1)

# Storage for results (much smaller than raw data)
pp_weighted_counts = np.zeros(num_pt_bins, dtype=np.float32)
aa_weighted_counts = np.zeros(num_pt_bins, dtype=np.float32)
bootstrap_samples_pp = []
bootstrap_samples_aa = []
batch_size = 1000

####################
# Hepdata business #
####################
def parse_hepdata_csv(filepath):
    """Parse a multi-block HEPData CSV file into a dict of DataFrames keyed by centrality."""
    datasets = {}
    current_centrality = None
    current_lines = []
    header = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            # Skip global metadata lines (before any centrality block)
            if line.startswith("#: Centrality"):
                # Save previous block if any
                if current_centrality is not None and current_lines:
                    block_text = "\n".join([header] + current_lines)
                    df = pd.read_csv(io.StringIO(block_text))
                    datasets[current_centrality] = df
                # Extract centrality label (last comma-separated field)
                current_centrality = line.split(",")[-1].strip()
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
    if current_centrality is not None and current_lines and header:
        block_text = "\n".join([header] + current_lines)
        df = pd.read_csv(io.StringIO(block_text))
        datasets[current_centrality] = df

    return datasets


def combine_uncertainties_asymmetric(stat_up, stat_dn, sys_up, sys_dn):
    """Combine statistical and systematic uncertainties in quadrature (asymmetric)."""
    total_up = np.sqrt(stat_up**2 + sys_up**2)
    total_dn = np.sqrt(stat_dn**2 + sys_dn**2)
    return total_up, total_dn

####################
# Compute $R_{AA}$ #
####################

for case in ["v", "m"]:
    hepmc_dir_case = hepmc_dir + case + "/"
    hepmc_files = os.listdir(hepmc_dir_case)

    weighted_counts_batch = []
    event_count = 0
    weight_sum = 0.0
    n_failed = 0

    num_files = len(hepmc_files)
    print(f"Number of {case} files: {num_files}")
    if max_files < num_files:
        print(f"Limiting to {max_files} files!")
    for file in hepmc_files[0:np.amin([max_files, num_files])]:
        try:
            with pyhepmc.open(hepmc_dir_case + file) as f:
                event = f.read()

            try:
                N_counts, _ = observables.hepmc_N(
                    hepmc_event=event,
                    rap_min=rapmin_RAA, rap_max=rapmax_RAA,
                    pTmin=pTmin_RAA, pTmax=pTmax_RAA, pT_bins=pT_bins,
                    include=pids
                )
            except:
                n_failed += 1
                continue
            weight = event.weight("pythia")

            weighted_counts_batch.append((N_counts * weight).astype(np.float32))
            event_count += 1
            weight_sum += weight

            # # Process batch
            # if event_count % batch_size == 0:
            #     if case == "v":
            #         bootstrap_samples_pp.append(np.array(weighted_counts_batch, dtype=np.float32))
            #     else:
            #         bootstrap_samples_aa.append(np.array(weighted_counts_batch, dtype=np.float32))
            #     weighted_counts_batch = []
            #
            del event

        except IsADirectoryError:
            continue

    # Handle remaining events
    # if weighted_counts_batch:
    weighted_counts_batch = np.array(weighted_counts_batch, dtype=np.float32)
    if case == "v":
        bootstrap_samples_pp.append(np.array(weighted_counts_batch/weight_sum, dtype=np.float32))
    else:
        bootstrap_samples_aa.append(np.array(weighted_counts_batch/weight_sum, dtype=np.float32))

    print(f"{n_failed} failed {case} events")

#################
# Bootstrapping #
#################

# Flatten batches for bootstrap
pp_data = np.concatenate(bootstrap_samples_pp, axis=0)
aa_data = np.concatenate(bootstrap_samples_aa, axis=0)
del bootstrap_samples_pp, bootstrap_samples_aa
gc.collect()

# Perform bootstrap on aggregated batches
rng = np.random.default_rng()

"""
Manual version
"""
def manual_bootstrap(data, n_resamples, rng):
    """
    Memory-efficient bootstrap: resample events, sum over events per bin.
    Returns distribution of shape (n_resamples, n_bins).
    """
    n_events = data.shape[0]
    dist = np.empty((n_resamples, data.shape[1]), dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_events, size=n_events)
        dist[i] = np.nanmean(data[idx], axis=0)
    return dist

print("Bootstrapping pp...")
pp_dist = manual_bootstrap(pp_data, n_bootstrap_samples, rng)  # (n_resamples, n_bins)
pp_avg = np.mean(pp_dist, axis=0)   # (n_bins,)
pp_err = np.std(pp_dist, axis=0)    # (n_bins,)
del pp_data, pp_dist
gc.collect()

print("Bootstrapping AA...")
aa_dist = manual_bootstrap(aa_data, n_bootstrap_samples, rng)  # (n_resamples, n_bins)
aa_avg = np.mean(aa_dist, axis=0)   # (n_bins,)
aa_err = np.std(aa_dist, axis=0)    # (n_bins,)
del aa_data, aa_dist
gc.collect()

"""
Scipy version
"""

# print("Bootstrapping pp...")
# pp_res = bootstrap((pp_data,), np.nansum, axis=0, confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
# pp_avg = np.mean(pp_res.bootstrap_distribution, axis=1)
# pp_err = pp_res.standard_error
# del pp_res
# gc.collect()
#
# print("Bootstrapping AA...")
# aa_res = bootstrap((aa_data,), np.nansum, axis=0, confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
# aa_avg = np.mean(aa_res.bootstrap_distribution, axis=1)
# aa_err = aa_res.standard_error
# del aa_res
# gc.collect()

RAA_avg = aa_avg / pp_avg
RAA_err = RAA_avg * np.sqrt((aa_err / aa_avg) ** 2 + (pp_err / pp_avg) ** 2)

############
# Plotting #
############

# Plot
pT_bin_cents = (pT_bins[1:] + pT_bins[0:-1]) / 2
fig = plt.figure(figsize=(8, 6))
axis = fig.add_subplot()

if plot_data:
    # ── Load data ──────────────────────────────────────────────────────────────────
    filepath = "HEPData-ins625472-v1-Figure_12a.csv"
    datasets = parse_hepdata_csv(filepath)

    colors = cm.plasma(np.linspace(0.05, 0.90, len(datasets)))

    centralities = ["0-10%", "30-40%", "60-70%"]
    for (centrality, df), color in zip(datasets.items(), colors):
        if centrality not in centralities:
            continue
        pt = df["$p_{T}$ (GeV/$c$)"].values
        raa = df["$R_{AA}$"].values
        stat_up = df["stat. +"].values
        stat_dn = np.abs(df["stat. -"].values)
        sys_up = df["sys. (bounds) +"].values
        sys_dn = np.abs(df["sys. (bounds) -"].values)

        # Combined uncertainties (stat ⊕ sys bounds) for error bars
        err_up, err_dn = combine_uncertainties_asymmetric(stat_up, stat_dn, sys_up, sys_dn)

        # pp systematic shown as a shaded band
        pp_up = df["sys. ($pp$) +"].values
        pp_dn = np.abs(df["sys. ($pp$) -"].values)

        axis.errorbar(
            pt, raa,
            yerr=[err_dn, err_up],
            fmt="o",
            color=color,
            markersize=4,
            linewidth=1.2,
            capsize=2,
            label=f"PHENIX {centrality}",
        )

        # pp normalization uncertainty as a semi-transparent band
        axis.fill_between(
            pt,
            raa - pp_dn,
            raa + pp_up,
            color=color,
            alpha=0.15,
        )

    # Reference line at R_AA = 1 (no suppression)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)


# Plot APE result on top
# axis.errorbar(pT_bin_cents, RAA_avg, yerr=RAA_err, marker='o', linestyle='-', color='r', label="APE 0-10%")
x = np.repeat(pT_bins, 2)[1:-1]        # shape (2N,)
y_mean  = np.repeat(RAA_avg, 2)
y_upper = np.repeat(RAA_avg + RAA_err, 2)
y_lower = np.repeat(RAA_avg - RAA_err, 2)
axis.fill_between(x, y_lower, y_upper, color='red', alpha=0.3,
                linewidth=0)
axis.plot(x, y_mean, color='red', lw=1.8, label=label)
axis.set_xlabel(r"$p_T$ (GeV/$c$)", fontsize=14)
axis.set_ylabel(r"$R_{AA}$", fontsize=14)
axis.set_ylim(0.0, 1.5)
axis.grid(True, linestyle=":", alpha=0.5)
axis.legend(fontsize=9, title_fontsize=10,
              loc="upper right", ncol=2, framealpha=0.85)
fig.savefig("RAA.png")
