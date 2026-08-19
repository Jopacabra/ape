import numpy as np
import timeit
import logging
import math

import matplotlib.pyplot as plt
import fastjet
import pyhepmc
import pythia8

import utilities



def EEC(jet: fastjet.PseudoJet=None, event: pythia8.Event=None, plot=False, pT_min=1, bins=np.linspace(-1, 1, 21)):
    """
    Function to compute energy-energy correlator for all particles in the event record or in a jet
    """

    # Access final state particles
    if event is not None:
        particles = [p for p in event if p.isFinal() and p.isCharged()]
    elif jet is not None:
        particles = [p for p in jet.constituents()]
    else:
        raise ValueError("Either event or fastjet_jet must be provided.")

    # Compute total energy
    total_energy = 0
    for i, p1 in enumerate(particles):
        total_energy += p1.e()

    # Conceptual EEC calculation
    cos_array = np.array([])
    EEC_array = np.array([])
    weight_array = np.array([])
    for i, p1 in enumerate(particles):
        # Enforce an energy cut on the first particle
        if p1.pt() < pT_min:
            continue
        for j, p2 in enumerate(particles):
            # Enforce an energy cut on the second particle
            if p2.pt() < pT_min:
                continue
            if i >= j: continue  # Avoid double counting

            # Calculate angle and energy weight -- use absolute value to set theta_2 > theta_1 -- exchanges i and j
            magnitudes = (np.sqrt((p1.px() ** 2) + (p1.py() ** 2) + (p1.pz() ** 2))
                          * np.sqrt((p2.px() ** 2) + (p2.py() ** 2) + (p2.pz() ** 2)))
            cos_theta = np.abs((p1.px() * p2.px() + p1.py() * p2.py() + p1.pz() * p2.pz()) / magnitudes)
            weight = (p1.e() * p2.e()) / (total_energy ** 2)
            EEC = cos_theta * weight

            # Add to arrays
            cos_array = np.append(cos_array, cos_theta)
            EEC_array = np.append(EEC_array, EEC)
            weight_array = np.append(weight_array, weight)

    # Bin the values of cos_array and compute average EEC
    bin_centers = (bins[:-1] + bins[1:])/2  # Compute bin centers

    # Calculate weighted sums and average
    hist_values, _ = np.histogram(cos_array, bins=bins, weights=weight_array)
    hist_weighted_EEC, _ = np.histogram(cos_array, bins=bins, weights=EEC_array * weight_array)
    average_EEC = np.nan_to_num(hist_weighted_EEC / hist_values, nan=0.0)  # Sets nans from hist_values=0 to be 0.0.

    if plot:
        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, average_EEC, marker='o', linestyle='-', color='b', label='Weighted Avg EEC')
        plt.xlabel('cos(theta)', fontsize=14)
        plt.ylabel('Average EEC', fontsize=14)
        plt.title('Energy-Energy Correlation (EEC) vs. cos(theta)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return average_EEC, bins

def hepmc_to_fastjet(hepmc_event: pyhepmc.GenEvent, R: float=0.4, scheme=fastjet.WTA_pt_scheme,
                     rap_min: float=0.0, rap_max: float=1.5, pTmin: float=0.0):
    """
    Function to run jetfinder on a HepMC event
    """
    # Access numpy interface of event object
    particles = hepmc_event.numpy.particles

    # Compute filter quantities
    E = particles.e
    px = particles.px
    py = particles.py
    pz = particles.pz
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # Protect denominator first
        denominator = np.where(E > pz, E - pz, np.finfo(float).tiny)
        rap = 0.5 * np.log((E + pz) / denominator)

    # # Handle remaining invalid values
    # rap[~np.isfinite(rap)] = np.inf  # or np.nan
    pT = np.hypot(px, py)

    # Filter
    ma = particles.status == 1  # Only consider final state particles
    # ma &= np.abs(particles.pid) == 211  # Only consider charged pions
    ma &= np.abs(rap) <= rap_max  # Cut on rapidity
    if rap_min > 0.0:
        ma &= np.abs(rap) > rap_min  # Cut on rapidity
    ma &= pT > pTmin  # Cut on transverse momentum

    # Iterate and make PseudoJets for fastjet.
    px = px[ma]
    py = py[ma]
    pz = pz[ma]
    E = E[ma]
    pseudo_particles = []
    for i in range(len(px)):
        # Append pseudojet for this little guy
        pseudo_particles.append(fastjet.PseudoJet(px[i], py[i], pz[i], E[i]))  # px, py, pz, E

    # Jet algorithm definition
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R, scheme)
    logging.info("Jetfinding using FastJet algorithm: {}".format(jet_def))

    # Find jets
    jets = jet_def(pseudo_particles)

    return jets

def hepmc_to_fastjet_gamma_jet_pairs(hepmc_event: pyhepmc.GenEvent, R: float=0.4, scheme=fastjet.WTA_pt_scheme,
                     rap_min: float=0.0, rap_max: float=1.5, pTmin: float=0.0, min_gamma_pt=0.0):
    """
    gamma_jet_finder.py

    Find gamma-jet pairs from HepMC files using pyhepmc and fastjet.

    Requirements:
        pip install pyhepmc fastjet numpy

    Physics strategy:
      1. Read each HepMC event.
      2. Find final-state (status==1) prompt photons (PDG ID 22).
         - In gamma-jet MC, the hard-scatter photon is typically the
           highest-pT photon. A status==1 photon NOT coming from a
           hadron decay is selected (we check the production vertex).
      3. Feed all other final-state charged+neutral hadrons (status==1,
           excluding neutrinos) into anti-kT jet clustering.
      4. Match the leading photon to the leading jet by requiring
           |Δφ| > π/2 (back-to-back in azimuth), as expected for
           direct gamma-jet pairs.
      5. Apply kinematic cuts appropriate for a heavy-ion measurement.
    """



    # ── PDG IDs to exclude from the jet input ──────────────────────────
    # Neutrinos (12, 14, 16) and their anti-particles are invisible.
    # Photons (22) are handled separately.
    INVISIBLE_PIDS = {12, -12, 14, -14, 16, -16}
    PHOTON_PID = 22

    # ── Analysis cuts ───────────────────────────────────────────────────
    # PHOTON_PT_MIN   = min_gamma_pt   # GeV  – minimum photon pT
    # PHOTON_ETA_MAX  = 0.9    # |η|  – ALICE-style barrel acceptance
    DPHI_MIN        = math.pi / 2.0   # back-to-back requirement -- minimum jet-photon acoplanarity

    # ── Jet definition ──────────────────────────────────────────────────
    jet_def  = fastjet.JetDefinition(fastjet.antikt_algorithm, R, scheme)


    # ── Helper: check if a photon is "prompt" ──────────────────────────
    def is_prompt_photon(particle) -> bool:
        """
        Return True if the photon does not come from a hadron decay.

        In Pythia HepMC output the production vertex parents tell
        us what produced the photon.  A photon whose parent is a quark,
        gluon, or the beam (no production vertex) is considered prompt.
        Photons from π⁰ → γγ have a π⁰ (pid=111) as parent.
        """
        vtx = particle.production_vertex
        for parent in vtx.particles_in:
            apid = abs(parent.pid)
            # Reject photons from light neutral mesons
            if apid in (111, 221, 331, 223, 113):   # π⁰, η, η', ω, ρ⁰
                return False
        return True


    # ── Helper: delta-phi in (−π, π] ───────────────────────────────────
    def delta_phi(phi1: float, phi2: float) -> float:
        dphi = phi1 - phi2
        while dphi >  math.pi: dphi -= 2 * math.pi
        while dphi < -math.pi: dphi += 2 * math.pi
        return dphi


    # ── Main event loop ─────────────────────────────────────────────────
    def find_gamma_jet_pairs(event):
        """
        Loop over events in *hepmc_file* and return a list of dicts,
        one per matched gamma-jet pair.
        """
        # results = []

        # ── 1. Collect final-state particles ───────────────────
        photons      = []
        jet_inputs   = []   # will become fastjet PseudoJets

        for p in event.particles:
            if p.status != 1:          # keep only stable final state
                continue
            mom = p.momentum           # FourVector: px, py, pz, e
            pid = p.pid
            apid = abs(pid)

            # Compute pT, η, φ from the four-momentum
            px, py, pz, e = mom.px, mom.py, mom.pz, mom.e
            pt = math.sqrt(px**2 + py**2)
            # if pt < 1e-6:
            #     continue
            if apid == PHOTON_PID:
                # ── Photon candidate ──────────────────────────
                # if (pt >= PHOTON_PT_MIN
                #         and abs(eta) <= PHOTON_ETA_MAX
                #         and is_prompt_photon(p)):
                #     # photons.append({
                #     #     "pt": pt, "eta": eta, "phi": phi,
                #     #     "px": px, "py": py, "pz": pz, "e": e,
                #     # })
                photons.append(p)  # will just take hardest one later

            if pt < pTmin:
                continue
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                # Protect denominator first
                denominator = np.where(e > pz, e - pz, np.finfo(float).tiny)
                eta = 0.5 * np.log((e + pz) / denominator)
            if np.abs(eta) < rap_min or np.abs(eta) > rap_max:
                continue
            # phi = math.atan2(py, px)


            elif apid not in INVISIBLE_PIDS:
                # ── Hadron / charged-particle input to jet finder
                # fastjet classic interface expects PseudoJet(px,py,pz,e)
                pj = fastjet.PseudoJet(px, py, pz, e)
                pj.set_user_index(len(jet_inputs))
                jet_inputs.append(pj)

        if not photons:
            # print("rejected all photons!")
            return None, None
        if not jet_inputs:
            # print("rejected all particle inputs!")
            return None, None


        # ── 2. Run jet finder ──────────────────────
        jets = fastjet.sorted_by_pt(jet_def(jet_inputs))

        if not jets:
            return None, None

        # # Apply jet η cut
        # jets = [j for j in jets if abs(j.eta()) <= JET_ETA_MAX]
        # if not jets:
        #     continue

        # ── 3. Select leading (highest-pT) photon ──────────────
        gamma = None
        for photon in photons:
            if gamma is None or photon.momentum.e > gamma.momentum.e:
                gamma = photon
        gamma_mom = gamma.momentum
        gamma = fastjet.PseudoJet(gamma_mom.px, gamma_mom.py, gamma_mom.pz, gamma_mom.e)  # Make a pseudojet from the gamma GenParticle
        if gamma is None:
            return None, None
        if math.sqrt(gamma_mom.px**2 + gamma_mom.py**2) < min_gamma_pt:
            return None, None

        # ── 4. Match photon to back-to-back jet ─────────────────
        # Take the leading jet that is back-to-back with the photon
        matched_jet = None
        for jet in jets:
            dphi = abs(delta_phi(gamma.phi(), jet.phi()))
            if dphi >= DPHI_MIN:
                matched_jet = jet
                break   # leading jet wins, we are already sorted by pT

        if matched_jet is None:
            return None, None

        # # ── 5. Store the pair kinematics ────────────────────────
        # pair = {
        #     # Photon
        #     "gamma_pt":       gamma["pt"],
        #     "gamma_eta":      gamma["eta"],
        #     "gamma_phi":      gamma["phi"],
        #     # Jet
        #     "jet_pt":         matched_jet.pt(),
        #     "jet_eta":        matched_jet.eta(),
        #     "jet_phi":        matched_jet.phi(),
        #     "jet_mass":       matched_jet.m(),
        #     "jet_nconst":     len(matched_jet.constituents()),
        #     # Pair observables
        #     "x_jgamma":       matched_jet.pt() / gamma["pt"],  # momentum balance
        #     "dphi":           abs(delta_phi(gamma["phi"], matched_jet.phi())),
        #     "deta":           matched_jet.eta() - gamma["eta"],
        # }
        # results.append(pair)
        #
        # print(
        #     f"Event {event_index:4d} | "
        #     f"γ  pT={gamma['pt']:.1f} η={gamma['eta']:+.2f} | "
        #     f"jet pT={matched_jet.pt():.1f} η={matched_jet.eta():+.2f} | "
        #     f"Δφ={pair['dphi']:.3f}  x_jγ={pair['x_jgamma']:.3f}"
        # )

        return gamma, matched_jet


    # ── Entry point ─────────────────────────────────────────────────────

    return find_gamma_jet_pairs(hepmc_event)

    # print(f"\nFound {len(pairs)} gamma-jet pairs.")

    # # Optional: dump to numpy / pandas for further analysis
    # if pairs:
    #     import pandas as pd
    #     df = pd.DataFrame(pairs)
    #     df.to_csv("gamma_jet_pairs.csv", index=False)
    #     print("Saved to gamma_jet_pairs.csv")
    #     print(df[["gamma_pt","jet_pt","x_jgamma","dphi"]].describe())

def fastjet_intrajetvnish_flow(jet: fastjet.PseudoJet=None, event: pythia8.Event=None, n=1, pT_min=0, flow="x",
                          E_bins=None):
    """
    Function to compute intrajet v_n harmonic with reference angle set to select on either x or z flow.
    """

    # Access final state particles
    if event is not None:
        particles = [p for p in event if p.isFinal() and p.isCharged()]
    elif jet is not None:
        particles = [p for p in jet.constituents()]
    else:
        raise ValueError("Either event or fastjet_jet must be provided.")

    # Get jet axis direction as unit vector
    jet_p = np.array([jet.px(), jet.py(), jet.pz()])
    jet_p = jet_p / np.linalg.norm(jet_p)

    # Get lists for particle properties
    phase = []
    pt_array = []

    # Compute observable
    if flow == "x":
        # Find the unit vector perpendicular to jet_p and z-axis.
        # alpha_0_vec = np.sign(jet_p[0]*jet_p[1])*np.cross(jet_p, np.array([0, 0, 1]))
        xHat = np.array([1, 0, 0])
        alpha_0_vec = np.sign(jet_p[0]) * (xHat - np.dot(xHat, jet_p) * jet_p)
    elif flow == "x2":
        zHat = np.array([0, 0, 1])
        alpha_0_vec = np.sign(jet_p[0] * jet_p[1]) * np.cross(jet_p, zHat)
    elif flow == "z":
        # Find the unit vector perpendicular to jet_p and x-axis.
        # alpha_0_vec = (-1) * np.sign(jet_p[2]*jet_p[1])*np.cross(jet_p, np.array([1, 0, 0]))
        zHat = np.array([0, 0, 1])
        alpha_0_vec = np.sign(jet_p[2]) * (zHat - np.dot(zHat, jet_p) * jet_p)
    elif flow == "z2":
        xHat = np.array([1, 0, 0])
        alpha_0_vec = -np.sign(jet_p[1] * jet_p[2]) * np.cross(jet_p, xHat)
    elif flow == "total":
        yHat = np.array([0, 1, 0])
        alpha_0_vec = np.sign(jet_p[1]) * (jet_p[1] * jet_p - yHat)
    else:
        # For "none" case, use a consistent lab-frame reference projected into perp plane
        # Later, we will not reference this vector in the vn calculation, we just use it to compute a consistent angle
        xHat = np.array([1, 0, 0])
        alpha_0_vec = xHat - np.dot(xHat, jet_p) * jet_p
        alpha_0_vec = alpha_0_vec / np.linalg.norm(alpha_0_vec)

    alpha_0_vec = alpha_0_vec / np.linalg.norm(alpha_0_vec)

    # Compute cosine(alpha) and E of each particle.
    total_weight = 0
    for i, p1 in enumerate(particles):
        p1_pt = p1.pt()
        if p1_pt < pT_min:
            continue
        p1_p = np.array([p1.px(), p1.py(), p1.pz()])  # Constituent particle direction
        p1_perp = utilities.perp_vec(p1_p, jet_p)  # Component of p1_p perpendicular to jet_p
        p1_norm = np.linalg.norm(p1_perp)
        if p1_norm == 0 or p1_norm == np.nan:
            # This particle aligns exactly with the jet axis, so we cannot compute a value. Skip!
            continue
        p1_perp = p1_perp / p1_norm  # Normalized unit vector

        if flow == "none":
            alpha = np.arccos(np.dot(alpha_0_vec, p1_perp))
            phase.append((np.exp(1j * n * alpha)))
        else:
            alpha = np.arccos(np.dot(alpha_0_vec, p1_perp))  # \vec{a}\cdot\vec{b} = ab \cos(\alpha), a = b = 1
            phase.append(np.cos(n * alpha))  # \cos(n * \alpha) phase factor for each particle
        pt_array.append(p1.pt())

    # Bin according to particle energy
    if E_bins is None:
        E_bins = np.linspace(pT_min, 15, 10)
    E_counts, _ = np.histogram(pt_array, bins=E_bins)
    phase_sum, _ = np.histogram(pt_array, bins=E_bins, weights=phase)

    # Compute averages and return
    with np.errstate(invalid='ignore'):
        avg_vns = phase_sum / E_counts  # Includes nans

    if flow == "none":
        avg_vns = np.absolute(avg_vns)
    return avg_vns, E_counts, E_bins

def fastjet_intrajetvnish_total(jet: fastjet.PseudoJet=None, event: pythia8.Event=None, n=1, pT_min=0, flow="x",
                                pt_weighting=0):
    """
    Function to compute intrajet v_n harmonic with reference angle set to select on either x or z flow.
    """

    # Access final state particles
    if event is not None:
        particles = [p for p in event if p.isFinal() and p.isCharged()]
    elif jet is not None:
        particles = [p for p in jet.constituents()]
    else:
        raise ValueError("Either event or fastjet_jet must be provided.")

    # Get jet axis direction as unit vector
    jet_p = np.array([jet.px(), jet.py(), jet.pz()])
    jet_p = jet_p / np.linalg.norm(jet_p)

    # Get lists for particle properties
    phase = []
    pt_array = []

    # Compute observable
    if flow == "x":
        # Find the unit vector perpendicular to jet_p and z-axis.
        # alpha_0_vec = np.sign(jet_p[0]*jet_p[1])*np.cross(jet_p, np.array([0, 0, 1]))
        xHat = np.array([1,0,0])
        alpha_0_vec = np.sign(jet_p[0]) * (xHat - np.dot(xHat, jet_p) * jet_p)
    elif flow == "x2":
        zHat = np.array([0,0,1])
        alpha_0_vec = np.sign(jet_p[0] * jet_p[1]) * np.cross(jet_p, zHat)
    elif flow == "z":
        # Find the unit vector perpendicular to jet_p and x-axis.
        # alpha_0_vec = (-1) * np.sign(jet_p[2]*jet_p[1])*np.cross(jet_p, np.array([1, 0, 0]))
        zHat = np.array([0,0,1])
        alpha_0_vec = np.sign(jet_p[2])*(zHat - np.dot(zHat, jet_p)*jet_p)
    elif flow == "z2":
        xHat = np.array([1,0,0])
        alpha_0_vec = -np.sign(jet_p[1] * jet_p[2]) * np.cross(jet_p, xHat)
    elif flow == "total":
        yHat = np.array([0,1,0])
        alpha_0_vec = np.sign(jet_p[1])*(jet_p[1]*jet_p - yHat)
    else:
        # For "none" case, use a consistent lab-frame reference projected into perp plane
        # Later, we will not reference this vector in the vn calculation, we just use it to compute a consistent angle
        xHat = np.array([1, 0, 0])
        alpha_0_vec = xHat - np.dot(xHat, jet_p) * jet_p
        alpha_0_vec = alpha_0_vec / np.linalg.norm(alpha_0_vec)

    alpha_0_vec = alpha_0_vec / np.linalg.norm(alpha_0_vec)

    # Compute cosine(alpha) and E of each particle.
    total_weight = 0
    skipped = 0
    for i, p1 in enumerate(particles):
        p1_pt = p1.pt()
        if p1_pt < pT_min:
            skipped += 1
            continue
        p1_p = np.array([p1.px(), p1.py(), p1.pz()])  # Constituent particle direction
        p1_perp = utilities.perp_vec(p1_p, jet_p)  # Component of p1_p perpendicular to jet_p
        p1_norm = np.linalg.norm(p1_perp)
        if p1_norm == 0 or p1_norm == np.nan:
            # This particle aligns exactly with the jet axis, so we cannot compute a value. Skip!
            continue
        p1_perp = p1_perp / p1_norm  # Normalized unit vector

        # Define an orthonormal frame in the plane perpendicular to jet_p
        alpha_0_vec_perp2 = np.cross(jet_p, alpha_0_vec)  # 90 degrees from alpha_0_vec, in the perp plane

        # Project p1_perp onto both basis vectors
        cos_component = np.dot(p1_perp, alpha_0_vec)  # "x" in the perp plane
        sin_component = np.dot(p1_perp, alpha_0_vec_perp2)  # "y" in the perp plane

        alpha = np.arctan2(sin_component, cos_component)  # angle in (-pi, pi]

        if flow == "none":
            phase.append((np.exp(1j * n * alpha)))
        else:
            phase.append(np.cos(n*alpha))  # \cos(n * \alpha) phase factor for each particle
        pt_array.append(p1.pt())

    # # Bin according to particle energy
    # if E_bins is None:
    #     E_bins = np.linspace(pT_min, 15, 10)
    # E_counts, _ = np.histogram(pt_array, bins=E_bins)
    # phase_sum, _ = np.histogram(pt_array, bins=E_bins, weights=phase)
    # print(f"Total: {len(particles)}, skipped: {skipped}, used: {len(phase)}")

    # Compute averages and return
    phase = np.array(phase)  # cast to numpy array for element-wise multiplication.
    with np.errstate(invalid='ignore'):

        pt_weight = np.array(pt_array) ** pt_weighting
        avg_vn = np.sum(phase * pt_weight) / np.sum(pt_weight) # Ignores nans

        # print(
        #     f"N={len(phase)}, phase[:5]={phase[:5]}, |Q|={np.abs(np.nansum(phase * pt_weight) / np.sum(pt_weight)):.4f}")

    if flow == "none":
        return np.absolute(avg_vn)
    else:
        return avg_vn


def fastjet_intrajetvnish_total_gammaref(jet: fastjet.PseudoJet=None, gamma: fastjet.PseudoJet=None,
                                         event: pythia8.Event=None, n=1, pT_min=0, flow="x",
                                         pt_weighting=0):
    """
    Function to compute intrajet v_n harmonic with reference angle set to select on either x or z flow, ABOUT GAMMA.
    """

    # Access final state particles
    if event is not None:
        particles = [p for p in event if p.isFinal() and p.isCharged()]
    elif jet is not None:
        particles = [p for p in jet.constituents()]
    else:
        raise ValueError("Either event or fastjet_jet must be provided.")

    # Get gamma axis direction as unit vector -- opposite to actual gamma_p
    gamma_p = (-1) * np.array([gamma.px(), gamma.py(), gamma.pz()])
    gamma_p = gamma_p / np.linalg.norm(gamma_p)

    # Get lists for particle properties
    phase = []
    pt_array = []

    # Compute observable
    if flow == "x":
        # Find the unit vector perpendicular to gamma_p and z-axis.
        # alpha_0_vec = np.sign(gamma_p[0]*gamma_p[1])*np.cross(gamma_p, np.array([0, 0, 1]))
        xHat = np.array([1, 0, 0])
        alpha_0_vec = np.sign(gamma_p[0]) * (xHat - np.dot(xHat, gamma_p) * gamma_p)
    elif flow == "x2":
        zHat = np.array([0, 0, 1])
        alpha_0_vec = np.sign(gamma_p[0] * gamma_p[1]) * np.cross(gamma_p, zHat)
    elif flow == "z":
        # Find the unit vector perpendicular to gamma_p and x-axis.
        # alpha_0_vec = (-1) * np.sign(gamma_p[2]*gamma_p[1])*np.cross(gamma_p, np.array([1, 0, 0]))
        zHat = np.array([0, 0, 1])
        alpha_0_vec = np.sign(gamma_p[2]) * (zHat - np.dot(zHat, gamma_p) * gamma_p)
    elif flow == "z2":
        xHat = np.array([1, 0, 0])
        alpha_0_vec = -np.sign(gamma_p[1] * gamma_p[2]) * np.cross(gamma_p, xHat)
    elif flow == "total":
        yHat = np.array([0, 1, 0])
        alpha_0_vec = np.sign(gamma_p[1]) * (gamma_p[1] * gamma_p - yHat)
    else:
        # For "none" case, use a consistent lab-frame reference projected into perp plane
        # Later, we will not reference this vector in the vn calculation, we just use it to compute a consistent angle
        xHat = np.array([1, 0, 0])
        alpha_0_vec = xHat - np.dot(xHat, gamma_p) * gamma_p
        alpha_0_vec = alpha_0_vec / np.linalg.norm(alpha_0_vec)

    alpha_0_vec = alpha_0_vec / np.linalg.norm(alpha_0_vec)

    # Compute cosine(alpha) and E of each particle.
    total_weight = 0
    skipped = 0
    for i, p1 in enumerate(particles):
        p1_pt = p1.pt()
        if p1_pt < pT_min:
            skipped += 1
            continue
        p1_p = np.array([p1.px(), p1.py(), p1.pz()])  # Constituent particle direction
        p1_perp = utilities.perp_vec(p1_p, gamma_p)  # Component of p1_p perpendicular to gamma_p
        p1_norm = np.linalg.norm(p1_perp)
        if p1_norm == 0 or p1_norm == np.nan:
            # This particle aligns exactly with the jet axis, so we cannot compute a value. Skip!
            continue
        p1_perp = p1_perp / p1_norm  # Normalized unit vector

        # Define an orthonormal frame in the plane perpendicular to gamma_p
        alpha_0_vec_perp2 = np.cross(gamma_p, alpha_0_vec)  # 90 degrees from alpha_0_vec, in the perp plane

        # Project p1_perp onto both basis vectors
        cos_component = np.dot(p1_perp, alpha_0_vec)  # "x" in the perp plane
        sin_component = np.dot(p1_perp, alpha_0_vec_perp2)  # "y" in the perp plane

        alpha = np.arctan2(sin_component, cos_component)  # angle in (-pi, pi]

        if flow == "none":
            phase.append((np.exp(1j * n * alpha)))
        else:
            phase.append(np.cos(n * alpha))  # \cos(n * \alpha) phase factor for each particle
        pt_array.append(p1.pt())

    # # Bin according to particle energy
    # if E_bins is None:
    #     E_bins = np.linspace(pT_min, 15, 10)
    # E_counts, _ = np.histogram(pt_array, bins=E_bins)
    # phase_sum, _ = np.histogram(pt_array, bins=E_bins, weights=phase)
    # print(f"Total: {len(particles)}, skipped: {skipped}, used: {len(phase)}")

    # Compute averages and return
    phase = np.array(phase)  # cast to numpy array for element-wise multiplication.
    with np.errstate(invalid='ignore'):

        pt_weight = np.array(pt_array) ** pt_weighting
        avg_vn = np.sum(phase * pt_weight) / np.sum(pt_weight)  # Ignores nans

        # print(
        #     f"N={len(phase)}, phase[:5]={phase[:5]}, |Q|={np.abs(np.nansum(phase * pt_weight) / np.sum(pt_weight)):.4f}")

    if flow == "none":
        return np.absolute(avg_vn)
    else:
        return avg_vn

# Signed acoplanarity in phi or eta
def signed_acoplanarity(jet: fastjet.PseudoJet=None, gamma: fastjet.PseudoJet=None, dir="phi"):
    """
    Function to compute signed acoplanarity in transverse or longitudinal direction
    """
    # First, compute the appropriate angles
    if dir == "phi":
        # Get phis
        jet_phi = np.mod(jet.phi(), 2*np.pi)  # Modulus to get on [0. 2pi)
        gamma_phi = np.mod(gamma.phi(), 2*np.pi)

        # Determine angle opposite to the gamma and its quadrant, thereby the sign
        gamma_aphi = np.mod(gamma_phi - np.pi, 2*np.pi)
        if gamma_aphi >= 0 and gamma_aphi < np.pi/2:
            # Quadrant 1
            sign = +1
        elif gamma_aphi >= np.pi/2 and gamma_aphi < np.pi:
            # Quadrant 2
            sign = -1
        elif gamma_aphi >= np.pi and gamma_aphi < 3*np.pi/2:
            # Quadrant 3
            sign = +1
        elif gamma_aphi >= 3*np.pi/2 and gamma_aphi < 2*np.pi:
            # Quadrant 4
            sign = -1

        return sign * (np.mod(jet.phi(), np.pi/2) - np.mod(gamma.phi(), np.pi/2))
    elif dir == "eta":
        # Get pseudorapidities
        jet_eta = jet.eta()
        gamma_eta = gamma.eta()

        if np.isnan(gamma_eta):
            return np.nan
        elif np.isnan(jet_eta):
            return np.nan
        elif gamma_eta == 0:
            return np.nan

        return np.sign(gamma_eta) * (jet_eta - gamma_eta)
    else:
        return None


# Signed deflection in phi or eta
def signed_deflection(jet1: fastjet.PseudoJet=None, jet2: fastjet.PseudoJet=None, dir="phi"):
    """
    Function to compute signed deflection in transverse or longitudinal direction

    If jet2 is more towards the attractor, make positive. If jet2 is farther from the attractor, make negative.
    """
    # First, compute the appropriate angles
    if dir == "phi":
        # Get phis
        jet1_phi = np.mod(jet1.phi(), 2 * np.pi)  # Modulus to get on [0. 2pi)
        # jet2_phi = np.mod(jet2.phi(), 2 * np.pi)

        # Determine quadrant of jet1, thereby the sign
        if jet1_phi >= 0 and jet1_phi < np.pi/2:
            # Quadrant 1
            sign = +1
        elif jet1_phi >= np.pi/2 and jet1_phi < np.pi:
            # Quadrant 2
            sign = -1
        elif jet1_phi >= np.pi and jet1_phi < 3*np.pi/2:
            # Quadrant 3
            sign = +1
        elif jet1_phi >= 3*np.pi/2 and jet1_phi < 2*np.pi:
            # Quadrant 4
            sign = -1

        return sign * (np.mod(jet1.phi(), np.pi / 2) - np.mod(jet2.phi(), np.pi / 2))
    elif dir == "eta":
        # Get pseudorapidities
        jet1_eta = jet1.eta()
        jet2_eta = jet2.eta()

        if np.isnan(jet1_eta):
            return np.nan
        elif np.isnan(jet2_eta):
            return np.nan
        elif jet1_eta == 0:
            return np.nan

        return np.sign(jet1_eta) * (jet2_eta - jet1_eta)

    else:
        return None



# Multiplicity calculation from hepmc file
def hepmc_N(hepmc_event: pyhepmc.GenEvent,
            rap_min: float=0.0, rap_max: float=1.5,
            pTmin: float=0.0, pTmax: float=100.0, pT_bins: np.ndarray=None,
            include=[211], exclude=None):
    """
    Function to compute multiplicity as a function of pT for a HepMC event
    """
    # Access numpy interface of event object
    particles = hepmc_event.numpy.particles

    # Compute filter quantities
    E = particles.e
    px = particles.px
    py = particles.py
    pz = particles.pz
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # Protect denominator first
        denominator = np.where(E > pz, E - pz, np.finfo(float).tiny)
        rap = 0.5 * np.log((E + pz) / denominator)
    pT = np.hypot(px, py)

    # Filter
    ma = particles.status == 1  # Only consider final state particles
    if include is not None:
        for ipid in include:
            ma &= np.abs(particles.pid) == ipid
    elif exclude is not None:
        for epid in exclude:
            ma &= np.abs(particles.pid) != epid
    ma &= np.abs(rap) <= rap_max  # Cut on rapidity
    if rap_min > 0.0:
        ma &= np.abs(rap) > rap_min  # Cut on rapidity
    ma &= pT > pTmin  # Cut on transverse momentum
    ma &= pT < pTmax  # Cut on transverse momentum

    # Cut pT array
    pT = pT[ma]

    # Histogram and return
    # Bin according to particle energy
    if pT_bins is None:
        pT_bins = np.linspace(pTmin, pTmax, 10)
    pT_counts, _ = np.histogram(pT, bins=pT_bins)

    return pT_counts, pT_bins


# Multiplicity calculation from hepmc file, sorted per pid
def hepmc_N_PID(hepmc_event: pyhepmc.GenEvent,
            rap_min: float=0.0, rap_max: float=1.5,
            pTmin: float=0.0, pTmax: float=100.0, pT_bins: np.ndarray=None,
            include=None, exclude=None):
    """
    Function to compute multiplicity as a function of pT for a HepMC event,
    returning a separate histogram for each unique particle type (by |pid|).

    Returns
    -------
    dict
        Keys are unique absolute PIDs (int). Values are tuples of
        (counts, bin_centers, pT_bins) — ready to pass directly to
        matplotlib's plt.bar / plt.step / plt.plot.
    """
    # Access numpy interface of event object
    particles = hepmc_event.numpy.particles

    # Compute filter quantities
    E = particles.e
    px = particles.px
    py = particles.py
    pz = particles.pz
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # Protect denominator first
        denominator = np.where(E > pz, E - pz, np.finfo(float).tiny)
        rap = 0.5 * np.log((E + pz) / denominator)
    pT = np.hypot(px, py)

    # Base filter: final state, rapidity window, pT window
    ma = particles.status == 1
    ma &= np.abs(rap) <= rap_max
    if rap_min > 0.0:
        ma &= np.abs(rap) > rap_min
    ma &= pT >= pTmin
    ma &= pT <= pTmax
    if include is not None:
        for ipid in include:
            ma &= np.abs(particles.pid) == ipid
    elif exclude is not None:
        for epid in exclude:
            ma &= np.abs(particles.pid) != epid

    # Build bin edges and centers once
    if pT_bins is None:
        pT_bins = np.linspace(pTmin, pTmax, 10)
    bin_centers = 0.5 * (pT_bins[:-1] + pT_bins[1:])

    # Apply base mask
    pT_masked  = pT[ma]
    pid_masked = np.abs(particles.pid[ma])

    # Histogram per unique PID
    results = {}
    for pid in np.unique(pid_masked):
        pid_sel = pid_masked == pid
        counts, _ = np.histogram(pT_masked[pid_sel], bins=pT_bins)
        results[int(pid)] = (counts, bin_centers, pT_bins)

    return results


def compute_event_Q_sum(hepmc_event, n, rap_min, rap_max, pTmin, pTmax, pT_bins, include):
    """
    Compute the per-pT-bin Q-vector components for harmonic n from a single
    HepMC event and the Pythia event weight.

    This function is essentially useless for all intents and purposes. Computing the vn at the level
    of a single event includes a huge host of non-flow v2-type correlations (dijets / gamma-jets).

    Returns
    -------
    Qx : ndarray, shape (n_bins,)   sum of cos(n*phi) for particles in each bin
    Qy : ndarray, shape (n_bins,)   sum of sin(n*phi) for particles in each bin
    M  : ndarray, shape (n_bins,)   particle multiplicity in each bin
    """
    n_bins = len(pT_bins) - 1
    Qx = np.zeros(n_bins, dtype=np.float64)
    Qy = np.zeros(n_bins, dtype=np.float64)
    M  = np.zeros(n_bins, dtype=np.float64)

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

    return Qx, Qy, M


def compute_event_Qs(hepmc_event, n, rap_min, rap_max, pTmin, pTmax, pT_bins, include):
    """
    Compute the per-particle per-pT-bin Q-vector components for harmonic n from a single
    HepMC event and the Pythia event weight.

    Returns
    -------
    Qx : ndarray, shape (n_bins,)   sum of cos(n*phi) for particles in each bin
    Qy : ndarray, shape (n_bins,)   sum of sin(n*phi) for particles in each bin
    M  : ndarray, shape (n_bins,)   particle multiplicity in each bin
    """
    n_bins = len(pT_bins) - 1
    Qx = [[] for _ in range(n_bins)]  # NOTE: use a list comprehension, not `[[]] * n_bins`!
    Qy = [[] for _ in range(n_bins)]
    M = [[] for _ in range(n_bins)]

    for particle in hepmc_event.particles:
        if particle.status != 1:  # final-state only
            continue
        pid = particle.pid
        if include and pid not in include:
            continue

        mom = particle.momentum
        pT = np.sqrt(mom.px ** 2 + mom.py ** 2)
        if pT < pTmin or pT >= pTmax:
            continue

        # rapidity
        try:
            rap = 0.5 * np.log((mom.e + mom.pz) / (mom.e - mom.pz))
        except (ZeroDivisionError, ValueError):
            continue
        if abs(rap) < rap_min or abs(rap) > rap_max:
            continue

        phi = np.arctan2(mom.py, mom.px)
        ibin = np.searchsorted(pT_bins[1:], pT)  # bin index
        ibin = min(ibin, n_bins - 1)

        Qx[ibin].append(np.cos(n * phi))
        Qy[ibin].append(np.sin(n * phi))
        M[ibin].append(1.0)

    # Array-ify them
    Qx = np.array(Qx)
    Qy = np.array(Qy)
    M = np.array(M)
    return Qx, Qy, M

