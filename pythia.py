import pythia8
import numpy as np
import pandas as pd
import config

# Function to generate a pp hard scattering at sqrt(s) = 5.02 TeV
def scattering(pThatmin=config.jet.PTHATMIN, pThatmax=config.jet.PTHATMAX, do_shower=config.jet.PROCESS_CORRECTIONS,
               type="dijet", min_pt=1):
    ############
    # Settings #
    ############
    y_res = 0.5
    soft_emission_cut = 0.1  # If do_shower, veto anything with fractional dijet pt difference > soft_emission_cut

    # Generate scattering id
    scattering_id = int(np.random.uniform(0, 1000000000000))

    ############################
    # Set up custom user hooks #
    ############################
    # Write own derived UserHooks class.
    class MyUserHooks(pythia8.UserHooks):

        # Constructor to make a user hook
        def __init__(self):
            pythia8.UserHooks.__init__(self)

        # Allow process cross section to be modified...
        def canModifySigma(self):
            return True

        # ...which gives access to the event at the trial level, before selection.
        def multiplySigmaBy(self, sigmaProcessPtr, phaseSpacePtr, inEvent):

            # All events should be 2 -> 2, kill them if not.
            if sigmaProcessPtr.nFinal() != 2: return 0.

            # Here we do not modify 2 -> 2 cross sections.
            return 1.

        # Allow a veto after process selection.
        def canVetoProcessLevel(self):
            return True

        # Veto events that do not fit the desired requirements
        def doVetoProcessLevel(self, process):
            # Get info
            info = pythia_process.infoPython()

            # Get only events at mid-rapidity, within my chosen y_res
            if np.abs(info.y()) < y_res:  # and np.abs(chosen_pt -np.abs(info.pTHat())) < pt_hat_res:
                return False  # Do not veto the event
            else:
                return True  # Veto the event

    #################
    # Set up Pythia #
    #################
    pythia_process = pythia8.Pythia("", False)  # Print header = False

    # Use seed based on time
    pythia_process.readString("Random:setSeed = on")
    pythia_process.readString("Random:seed = 0")

    # # Set beam energy - in GeV
    pythia_process.readString("Beams:eCM = {}".format(config.constants.ROOT_S))

    # Set particles in each beam - defaults to proton (2212), if nothing set
    pythia_process.readString("Beams:idA = 2212")
    pythia_process.readString("Beams:idB = 2212")

    # Do the hard scattering process level stuff
    pythia_process.readString("ProcessLevel:all = on")

    # # Choose what mode for the results
    # if do_shower:
    #     # Enable parton-level interactions -- things will happen after the hard scattering.
    #     pythia_process.readString("PartonLevel:all = on")
    #
    #     # Turn off multi-parton interactions
    #     pythia_process.readString("PartonLevel:MPI = off")
    #
    #     # Turn off "Initial state" radiation with spacelike particles "before the hard process"
    #     pythia_process.readString("PartonLevel:ISR = off")
    #     # pythia_process.readString("PartonLevel:MPI = {}".format(do_shower))  # Multi-parton interactions
    #     # pythia_process.readString("PartonLevel:ISR = {}".format(
    #     #     do_shower))  # "Initial state" radiation with spacelike particles "before the hard process"
    #     pythia_process.readString("PartonLevel:FSR = off")  # "Final state" radiation with timelike particles
    #     pythia_process.readString("PartonLevel:Remnants = off")  # Turn off beam remnant adding
    #     pythia_process.readString("Check:event = off") # Turn off event checks -- Missing beam remnants!
    # else:
    #     pythia_process.readString("PartonLevel:all = off")
    pythia_process.readString("PartonLevel:all = off")  # No showering business implemented yet.

    # Only parton-level results, no hadronization
    pythia_process.readString("HadronLevel:all = off")

    # Choose appropriate hard processes
    if type == "dijet":  # Light quark and gluon dijets
        # Processes that yield light quarks and gluons
        pythia_process.readString("HardQCD:gg2gg = on")
        pythia_process.readString("HardQCD:gg2qqbar = on")
        pythia_process.readString("HardQCD:qg2qg = on")
        pythia_process.readString("HardQCD:qq2qq = on")
        pythia_process.readString("HardQCD:qqbar2gg = on")
        pythia_process.readString("HardQCD:qqbar2qqbarNew = on")

    elif type == "gamma-jet":
        pythia_process.readString("PromptPhoton:qg2qgamma = on")
        pythia_process.readString("PromptPhoton:qqbar2ggamma = on")
        pythia_process.readString("PromptPhoton:gg2ggamma = on")

    # There are some other process types we should consider for other purposes.
    # pythia_process.readString("HardQCD:3parton = on")  # 3 parton kinds of processes...
    # pythia_process.readString("HardQCD:gg2ccbar = on")  # heavy quark kinds of processes...

    # Set a phase space cut for particle pT.
    '''
    The HardQCD 2->2 processes are divergent as pT -> 0, so we need some cut here.
    Note that this parton-level cut does not necessarily put a cut on jet phase space.
    intermediate parton showers, MPIs, hadronization effects, and jet finders will distort the original simple process
    '''
    pythia_process.readString("PhaseSpace:pTHatMin = {}".format(pThatmin))  # Phase space cuts are on hard process pTHat
    pythia_process.readString("PhaseSpace:pTHatMax = {}".format(pThatmax))

    # Here we bias the selection of pTHat for the process by a given power of pTHat (Here pTHat^4)
    # This is more or less equivalent to sampling from a uniform distribution in pTHat
    # and recording an appropriate true pTHat-dependent weight from a known weight distribution
    pythia_process.readString("PhaseSpace:bias2Selection = on")
    pythia_process.readString("PhaseSpace:bias2SelectionPow = 4")

    # Set up to do a user veto and send it in.
    myUserHooks = MyUserHooks()
    pythia_process.setUserHooksPtr(myUserHooks)

    # Tell Pythia to "do the thing" (run with the configurations above)
    pythia_process.init()

    # Event loop. Iterate until getting a satisfactory hard process.
    for iEvent in range(1000):
        # Run the next event
        if not pythia_process.next(): continue
        nCharged = 0
        max_0 = 0
        max_0_i = 0
        max_1 = 0
        max_1_i = 0

        # Check if we're using the event record or the process record
        if pythia_process.event.size() == 0:
            num_particles_hist = pythia_process.process.size()
            record = pythia_process.process
        else:
            num_particles_hist = pythia_process.event.size()
            record = pythia_process.event

        # Iterate over all particles and look for the hardest partonic outputs.
        for i in np.arange(0, num_particles_hist):
            p = record[i]
            if (p.status() > 0):  # and p.isHadron() and p.isCharged():
                nCharged += 1
                if p.pT() > max_1:
                    if p.pT() > max_0:
                        max_0_i = i
                        max_0 = p.pT()
                    else:
                        max_1_i = i
                        max_1 = p.pT()
        ids = [record[max_0_i].id(), record[max_1_i].id()]
        ys = [record[max_0_i].y(), record[max_1_i].y()]

        # Check if the results are The correct flavors, above pt cut, and in rapidity cut
        if type == "dijet":
            if ((np.abs(ids[0]) < 3.1) or (np.abs(ids[0]) == 21)) and ((np.abs(ids[1]) < 3.1) or (np.abs(ids[1]) == 21)):
                if (max_0 > min_pt) and (max_1 > min_pt):
                    if np.abs(ys[0]) < y_res and np.abs(ys[1]) < y_res:
                        if ((max_0 - max_1)/max_0 < soft_emission_cut):
                            break  # Stop generating events, keep these particles

        if type == "gamma-jet":
            if (((np.abs(ids[0]) < 3.1) or (np.abs(ids[0]) == 21) or (np.abs(ids[0]) == 22))
                    and ((np.abs(ids[1]) < 3.1) or (np.abs(ids[1]) == 21) or (np.abs(ids[1]) == 22))):
                if ids[0] == 22 or ids[1] == 22:  # At least one photon
                    if (max_0 > min_pt) and (max_1 > min_pt):
                        if np.abs(ys[0]) < y_res and np.abs(ys[1]) < y_res:
                            if ((max_0 - max_1)/max_0 < soft_emission_cut):
                                break  # Stop generating events, keep these particles

    ################################
    # Package and output particles #
    ################################
    weight = pythia_process.infoPython().weight()
    particles = pd.DataFrame({})

    for particle in [record[max_0_i], record[max_1_i]]:
        properties = pd.DataFrame(
            {
                'id': [int(particle.id())],
                'status': [int(particle.status())],
                'mother1': [int(particle.mother1())],
                'mother2': [int(particle.mother2())],
                'daughter1': [int(particle.daughter1())],
                'daughter2': [int(particle.daughter2())],
                'col': [int(particle.col())],
                'acol': [int(particle.acol())],
                'px': [particle.px()],
                'py': [particle.py()],
                'pz': [particle.pz()],
                'pt': [particle.pT()],
                'y': [particle.y()],
                'e': [particle.e()],
                'm': [particle.m()],
                'scaleIn': [particle.scale()]
            }
        )

        particles = pd.concat([particles, properties], axis=0)

    return particles, weight


# Function to hadronize a pair of particles
def string_hadronize(jet1, jet2, scaleIn=2, weight=1):
    # Settings
    y_res = 1
    max_had_runs = 10000

    #########################
    # Assign colors to jets #
    #########################

    # Get particle ids
    id1 = jet1.id
    id2 = jet2.id

    remnant = False
    remnant2 = False
    # Choose colors so as to get a color singlet
    # Add a third particle as a beam remnant to get a color singlet, if necessary
    if id1 == 21 and id2 == 21:
        # A pair of gluons
        # Particles just get opposite colors and anticolors
        col1 = 101
        acol1 = 102
        col2 = 102
        acol2 = 101
    elif (3.1 > id1 > 0) and (-3.1 < id2 < 0):
        # Quark antiquark pair
        col1 = 101
        acol1 = 0
        col2 = 0
        acol2 = 101
    elif (-3.1 < id1 < 0) and (3.1 > id2 > 0):
        # Antiquark quark pair
        col1 = 0
        acol1 = 101
        col2 = 101
        acol2 = 0
    elif (-3.1 < id1 < 0) and id2 == 21:
        # antiquark gluon pair
        remnant = True
        rem_col = 102
        rem_acol = 0
        col1 = 0
        acol1 = 101
        col2 = 101
        acol2 = 102
    elif id1 == 21 and (-3.1 < id2 < 0):
        # gluon antiquark pair
        remnant = True
        rem_col = 102
        rem_acol = 0
        col1 = 101
        acol1 = 102
        col2 = 0
        acol2 = 101
    elif (3.1 > id1 > 0) and id2 == 21:
        # quark gluon pair
        remnant = True
        rem_col = 0
        rem_acol = 102
        col1 = 101
        acol1 = 0
        col2 = 102
        acol2 = 101
    elif id1 == 21 and (3.1 > id2 > 0):
        # gluon quark pair
        remnant = True
        rem_col = 0
        rem_acol = 101
        col1 = 101
        acol1 = 102
        col2 = 102
        acol2 = 0
    elif (3.1 > id1 > 0) and (3.1 > id2 > 0):
        # quark quark pair
        remnant = True
        rem_col = 0
        rem_acol = 101
        rem2_col = 0
        rem2_acol = 102
        col1 = 101
        acol1 = 0
        col2 = 102
        acol2 = 0
    elif (-3.1 < id1 < 0) and (-3.1 < id2 < 0):
        # antiquark antiquark pair
        remnant = True
        rem_col = 101
        rem_acol = 0
        rem2_col = 102
        rem2_acol = 0
        col1 = 0
        acol1 = 101
        col2 = 0
        acol2 = 102

    col_array = np.array([col1, col2])
    acol_array = np.array([acol1, acol2])
    if remnant:
        col_array = np.append(col_array, rem_col)
        acol_array = np.append(acol_array, rem_acol)
    if remnant2:
        col_array = np.append(col_array, rem2_col)
        acol_array = np.append(acol_array, rem2_acol)

    ############################################
    # Set up Pythia instance for hadronization #
    ############################################

    # Instantiate Pythia
    pythia_had = pythia8.Pythia("", False)  # Print header = False

    # Use seed based on time
    pythia_had.readString("Random:setSeed = on")
    pythia_had.readString("Random:seed = 0")

    # Only do the hadron level stuff
    pythia_had.readString("ProcessLevel:all = off")
    pythia_had.readString("PartonLevel:all = off")
    pythia_had.readString("HadronLevel:all = on")

    # Don't allow pi^0 to decay:
    pythia_had.readString("111:mayDecay = off")

    # Allow color reconnection in hadronization
    # pythia_had.readString("ColourReconnection:forceHadronLevelCR = on")

    # Turn off event checks that enforce conservation of momentum in the event
    pythia_had.readString("Check:event = on")

    # Tell Pythia to "do the thing" (run with the configurations above)
    pythia_had.init()

    #################################
    # Run the hadronization routine #
    #################################
    # We repeatedly hadronize until we come out with a satisfactory pion, saving info on the statistical weight
    accepted = False
    total_pions = 0
    total_had_runs = 0
    success_had_runs = 0
    while total_had_runs < max_had_runs:

        # Clear the event
        pythia_had.event.reset()

        # Add in edited particles
        part_i = -1
        i = 0
        # jet 1
        pythia_had.event.append(id=int(id1), status=int(23),
                                col=int(col_array[0]), acol=int(acol_array[0]),
                                px=float(jet1.p_x), py=float(jet1.p_y), pz=0,
                                e=float(np.sqrt(jet1.p_x**2 + jet1.p_y**2 + float(jet1.m)**2)),
                                m=float(jet1.m),
                                scaleIn=float(scaleIn))
        i += 1
        # jet 2
        pythia_had.event.append(id=int(id2), status=int(23),
                                col=int(col_array[1]), acol=int(acol_array[1]),
                                px=float(jet2.p_x), py=float(jet2.p_y), pz=0,
                                e=float(np.sqrt(jet2.p_x ** 2 + jet2.p_y ** 2 + float(jet2.m) ** 2)),
                                m=float(jet2.m),
                                scaleIn=float(scaleIn))
        i += 1

        if remnant:
            if rem_col != 0:
                rem_id = np.random.default_rng().choice([2, 2, 1])
                if rem_id == 2:
                    rem_m = 0.0022
                else:
                    rem_m = 0.0047
            else:
                rem_id = np.random.default_rng().choice([-2, -2, -1])
                if rem_id == -2:
                    rem_m = 0.0022
                else:
                    rem_m = 0.0047
            pythia_had.event.append(id=int(rem_id), status=int(23),
                                    col=int(col_array[2]), acol=int(acol_array[2]),
                                    px=0, py=0, pz=10000,
                                    e=float(np.sqrt(0 ** 2 + 0 ** 2 + 10000 **2 + rem_m ** 2)),
                                    m=float(rem_m),
                                    scaleIn=float(scaleIn))
            i += 1
        if remnant2:
            if rem2_col != 0:
                rem2_id = np.random.default_rng().choice([2, 2, 1])
                if rem2_id == 2:
                    rem2_m = 0.0022
                else:
                    rem2_m = 0.0047
            else:
                rem2_id = np.random.default_rng().choice([-2, -2, -1])
                if rem2_id == -2:
                    rem2_m = 0.0022
                else:
                    rem2_m = 0.0047
            pythia_had.event.append(id=int(rem2_id), status=int(23),
                                    col=int(col_array[3]), acol=int(acol_array[3]),
                                    px=0, py=0, pz=-10000,
                                    e=float(np.sqrt(0 ** 2 + 0 ** 2 + 10000 **2 + rem2_m ** 2)),
                                    m=float(rem2_m),
                                    scaleIn=float(scaleIn))
            i += 1
        # part_i = -1
        # for particle in pythia_had.process:
        #     part_i += 1
        # # Force parton shower
        # # Set all particles allowed to shower
        # shower_pTmax = pythia_process.infoPython().pTHat()
        # pythia_had.forceTimeShower(iBeg=part_i-1, iEnd=part_i, pTmax=shower_pTmax)  #, nBranchMax=10)

        # List particles for debug
        pythia_had.event.list()

        # hadronize - restart if event checks fail
        if not pythia_had.next():
            total_had_runs += 1  # Add a total hadronization
            continue
        success_had_runs += 1  # Add a successful hadronization
        total_had_runs += 1  # Add a total hadronization

        # List particles again for debug
        pythia_had.event.list()

        # Look for an acceptable pion
        hadron_accepted_px = np.array([])
        hadron_accepted_py = np.array([])
        hadron_accepted_pz = np.array([])
        hadron_accepted_y = np.array([])
        hadron_accepted_e = np.array([])
        hadron_accepted_pt = np.array([])
        hadron_accepted_id = np.array([])
        hadron_f_pt = np.array([])
        pions_f = 0
        for particle in pythia_had.event:
            if particle.status() > 0:  # Particle exists in the final state
                id = particle.id()
                if id == 111 or np.abs(id) == 211:  # Collect pions
                    pions_f += 1
                    hadron_y = particle.y()
                    hadron_pt = particle.pT()
                    hadron_f_pt = np.append(hadron_f_pt, np.abs(hadron_pt))
                    pions_f += 1
                    if np.abs(hadron_y) < 1:  # Particle is at mid-rapidity
                        if np.abs(hadron_pt) > 1:  # Particle is hard -- substantially above medium scale
                            accepted = True
                            hadron_accepted_px = np.append(hadron_accepted_px, particle.px())
                            hadron_accepted_py = np.append(hadron_accepted_py, particle.py())
                            hadron_accepted_pz = np.append(hadron_accepted_pz, particle.pz())
                            hadron_accepted_y = np.append(hadron_accepted_y, hadron_y)
                            hadron_accepted_e = np.append(hadron_accepted_e, particle.e())
                            hadron_accepted_pt = np.append(hadron_accepted_pt, np.abs(particle.pT()))
                            hadron_accepted_id = np.append(hadron_accepted_id, particle.id())

        # Count pions and runs to determine weight of the final pion
        total_pions += pions_f

        if accepted:
            break


    hadrons = pd.DataFrame(
        {
            'id': hadron_accepted_id.astype(int),
            'px': hadron_accepted_px.astype(float),
            'py': hadron_accepted_py.astype(float),
            'pz': hadron_accepted_pz.astype(float),
            'pt': hadron_accepted_pt.astype(float),
            'y': hadron_accepted_y.astype(float),
            'e': hadron_accepted_e.astype(float),
            'weight': np.full_like(hadron_accepted_id, float(weight)).astype(float),
            'num_hrz': np.full_like(hadron_accepted_id, int(success_had_runs)).astype(int),
            'failures': np.full_like(hadron_accepted_id, int(total_had_runs - success_had_runs)).astype(int)
        })

    return hadrons


# Function to hadronize a list of particles already including colors and anticolors and get pythia event
def pp_shower_hadronize(particles):
    # Settings
    y_res = 1
    max_had_runs = 10000

    ############################################
    # Set up Pythia instance for hadronization #
    ############################################

    # Instantiate Pythia
    pythia_had = pythia8.Pythia("", False)  # Print header = False

    # Use seed based on time
    pythia_had.readString("Random:setSeed = on")
    pythia_had.readString("Random:seed = 0")

    # Only do the hadron level stuff
    pythia_had.readString("ProcessLevel:all = off")
    pythia_had.readString("PartonLevel:all = off")
    pythia_had.readString("HadronLevel:all = on")

    # Don't allow pi^0 to decay:
    pythia_had.readString("111:mayDecay = off")

    # Allow color reconnection in hadronization
    # pythia_had.readString("ColourReconnection:forceHadronLevelCR = on")

    # Turn off event checks that enforce conservation of momentum in the event
    pythia_had.readString("Check:event = on")

    # Tell Pythia to "do the thing" (run with the configurations above)
    pythia_had.init()

    #################################
    # Run the hadronization routine #
    #################################
    # We repeatedly hadronize until we come out with a satisfactory pion, saving info on the statistical weight
    accepted = False
    total_pions = 0
    total_had_runs = 0
    success_had_runs = 0
    while total_had_runs < max_had_runs:

        # Clear the event
        pythia_had.event.reset()

        # Add in edited particles
        for index, particle in particles.iterrows():
            pythia_had.event.append(id=int(particle['id']), status=int(particle['status']),
                                    col=int(particle['col']), acol=int(particle['acol']),
                                    px=float(particle['px']), py=float(particle['py']), pz=float(particle['pz']),
                                    e=float(np.sqrt(float(particle['px']) ** 2 + float(particle['py']) ** 2
                                                    + float(particle['pz']) **2 + float(particle['m']) ** 2)),
                                    m=float(particle['m']),
                                    scaleIn=float(particle['scaleIn']))


        # part_i = -1
        # for particle in pythia_had.process:
        #     part_i += 1
        # # Force parton shower
        # # Set all particles allowed to shower
        # shower_pTmax = pythia_process.infoPython().pTHat()
        # pythia_had.forceTimeShower(iBeg=part_i-1, iEnd=part_i, pTmax=shower_pTmax)  #, nBranchMax=10)

        # List particles for debug
        pythia_had.event.list()

        # hadronize - restart if event checks fail
        if not pythia_had.next():
            total_had_runs += 1  # Add a total hadronization
            continue
        success_had_runs += 1  # Add a successful hadronization
        total_had_runs += 1  # Add a total hadronization

        # List particles again for debug
        pythia_had.event.list()

    return pythia_had.event
