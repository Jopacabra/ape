import pyhepmc
import os
import sys
import gc
import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
import observables

# results subdirectory for HepMC files
name = "hepmc"

# Statistics settings
n_bootstrap_samples = 1000

# Particle cuts
pTmin_RAA = 2.0
pTmax_RAA = 100
rapmin_RAA = 0.0
rapmax_RAA = 1000
num_pt_bins = 15
pT_bins = np.linspace(pTmin_RAA, pTmax_RAA, num_pt_bins+1)

# Storage for results (much smaller than raw data)
pp_weighted_counts = np.zeros(num_pt_bins, dtype=np.float32)
aa_weighted_counts = np.zeros(num_pt_bins, dtype=np.float32)
bootstrap_samples_pp = []
bootstrap_samples_aa = []
batch_size = 100

for case in ["v", "m"]:
    hepmc_dir = f"../results/{name}/{case}/"
    hepmc_files = os.listdir(hepmc_dir)

    weighted_counts_batch = []
    event_count = 0

    print(f"Number of {case} files: {len(hepmc_files)}")
    for file in hepmc_files:
        try:
            with pyhepmc.open(hepmc_dir + file) as f:
                event = f.read()

            N_counts, _ = observables.hepmc_N(
                hepmc_event=event,
                rap_min=rapmin_RAA, rap_max=rapmax_RAA,
                pTmin=pTmin_RAA, pTmax=pTmax_RAA, pT_bins=pT_bins
            )
            weight = event.weight("pythia")

            weighted_counts_batch.append((N_counts * weight).astype(np.float32))
            event_count += 1

            # Process batch
            if event_count % batch_size == 0:
                if case == "v":
                    bootstrap_samples_pp.append(np.array(weighted_counts_batch))
                else:
                    bootstrap_samples_aa.append(np.array(weighted_counts_batch))
                weighted_counts_batch = []

            del event

        except IsADirectoryError:
            continue

    # Handle remaining events
    if weighted_counts_batch:
        if case == "v":
            bootstrap_samples_pp.append(np.array(weighted_counts_batch, dtype=np.float32))
        else:
            bootstrap_samples_aa.append(np.array(weighted_counts_batch, dtype=np.float32))

# Perform bootstrap on aggregated batches
rng = np.random.default_rng()

# Flatten batches for bootstrap

pp_data = np.concatenate(bootstrap_samples_pp, axis=0)
aa_data = np.concatenate(bootstrap_samples_aa, axis=0)
del bootstrap_samples_pp, bootstrap_samples_aa
gc.collect()

print("Bootstrapping pp...")
pp_res = bootstrap((pp_data,), np.nansum, axis=0, confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
pp_avg = np.mean(pp_res.bootstrap_distribution, axis=1)
pp_err = pp_res.standard_error
del pp_res
gc.collect()

print("Bootstrapping AA...")
aa_res = bootstrap((aa_data,), np.nansum, axis=0, confidence_level=0.9, rng=rng, n_resamples=n_bootstrap_samples)
aa_avg = np.mean(aa_res.bootstrap_distribution, axis=1)
aa_err = aa_res.standard_error
del aa_res
gc.collect()

RAA_avg = aa_avg / pp_avg
RAA_err = RAA_avg * np.sqrt((aa_err / aa_avg) ** 2 + (pp_err / pp_avg) ** 2)


# Plot
pT_bin_cents = (pT_bins[1:] + pT_bins[0:-1]) / 2
fig = plt.figure(figsize=(8, 6))
axis = fig.add_subplot()
axis.errorbar(pT_bin_cents, RAA_avg, yerr=RAA_err, marker='o', linestyle='-', color='r')
# axis.errorbar(E_bin_cents, v2_avg, yerr=v2_avg_err, marker='x', linestyle='--', color='r', label=flow + r' $\Delta v_2$')
# axis.errorbar(E_bin_cents, v3_avg, yerr=v3_avg_err, marker='^', linestyle=':', color='r', label=flow + r' $\Delta v_3$')
axis.set_xlabel(r'$p_T$ (GeV)', fontsize=14)
axis.set_ylabel(r'$R_{AA}$', fontsize=14)
# axis.legend(fontsize=12)
axis.set_ylim(0.0, 1.5)
# plt.set_xscale("log")
# plt.set_yscale("log")
axis.grid(True)
fig.savefig("RAA.png")
