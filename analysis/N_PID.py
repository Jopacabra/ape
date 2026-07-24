import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyhepmc

sys.path.insert(0, '..')
import observables

# ── Configuration ──────────────────────────────────────────────────────────────
hepmc_dir = "../results/hepmc/m/"   # directory of HepMC files
output_file = "N_PID_m.png"

# Particle cuts
pTmin = 0.5
pTmax = 20.0
rap_min = 0.0
rap_max = 5
num_pt_bins = 20
pT_bins = np.linspace(pTmin, pTmax, num_pt_bins + 1)
bin_centers = 0.5 * (pT_bins[:-1] + pT_bins[1:])

# PIDs to include (e.g. neutrinos) -- these will be counted or flagged -- takes precedence over exclude
include = None  # [211]  # or None
# OR PIDs to exclude (e.g. neutrinos) -- these will NOT be counted or flagged
exclude = [22]  # [12, 14, 16, 22]  # or None
# Flag any events with these particles!
flag = [1, 2, 3, 4, 5, 6, 7, 8, 21]
# ──────────────────────────────────────────────────────────────────────────────

# Accumulate counts per PID across all events.
# combined_counts: dict mapping pid (int) -> np.ndarray of shape (num_pt_bins,)
combined_counts = {}
n_events = 0
n_failed = 0
flagged = 0

hepmc_files = sorted(os.listdir(hepmc_dir))
print(f"Found {len(hepmc_files)} files in {hepmc_dir}")

for fname in hepmc_files:
    fpath = os.path.join(hepmc_dir, fname)
    if not os.path.isfile(fpath):
        continue

    try:
        with pyhepmc.open(fpath) as f:
            event = f.read()
    except Exception as e:
        print(f"  Could not open {fname}: {e}")
        n_failed += 1
        continue

    if event is None:
        n_failed += 1
        continue

    flag_this = False
    weight = event.weight("pythia")

    try:
        pid_hists = observables.hepmc_N_PID(
            hepmc_event=event,
            rap_min=rap_min,
            rap_max=rap_max,
            pTmin=pTmin,
            pTmax=pTmax,
            pT_bins=pT_bins,
            exclude=exclude,
            include=include,
        )
    except Exception as e:
        print(f"  Observable failed for {fname}: {e}")
        n_failed += 1
        continue

    # Accumulate: PIDs missing from this event simply contribute zero counts,
    # which is handled automatically by only adding where the PID appears.
    for pid, (counts, _bin_centers, _pT_bins) in pid_hists.items():
        if pid in flag:
            flag_this = True
        if pid not in combined_counts:
            # First time we see this PID — initialise with zeros so earlier
            # events (which didn't contain it) contribute nothing.
            combined_counts[pid] = np.zeros(num_pt_bins, dtype=np.float64)
        combined_counts[pid] += weight*counts

    if flag_this:
        flagged += 1

    n_events += 1

print(f"Processed {n_events} events ({n_failed} failed/skipped).")
print(f"Found {len(combined_counts)} unique |PID| values: {sorted(combined_counts.keys())}")
print(f"Flagged {flagged} events for quarks or gluons.")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

# Sort PIDs so the legend is ordered
for pid in sorted(combined_counts.keys()):
    counts = combined_counts[pid]
    if counts.sum() == 0:
        continue  # skip PIDs with no entries after all cuts
    ax.step(pT_bins[:-1], counts, where="post", linewidth=3, label=f"|PID| = {pid}")

ax.set_xlabel(r"$p_T$ (GeV)", fontsize=13)
ax.set_ylabel("Total counts", fontsize=13)
ax.set_title(
    f"Particle multiplicity vs $p_T$ per species\n"
    f"{n_events} events, "
    r"$|y| \in $" + f"[{rap_min}, {rap_max}], "
    r"$p_T \in $" + f"[{pTmin}, {pTmax}] GeV",
    fontsize=11,
)
ax.set_yscale("log")
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Saved plot to {output_file}")
plt.show()