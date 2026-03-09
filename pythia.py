import logging

import pythia8
import numpy as np
import fastjet

import config
import hard_particles

# Function to generate a pp hard scattering
def scattering(pThatmin=config.jet.PTHATMIN, pThatmax=config.jet.PTHATMAX, do_shower=config.jet.SHOWER,
               type="dijet", min_pt=1, get_all=True, tau=config.transport.hydro.TAU_FS, x=0, y=0, etas=0,
               y_res = config.jet.RAP_MAX):
    ############
    # Settings #
    ############
    soft_emission_cut = 0.1  # For parton pairs, veto anything with fractional diparton pt difference > soft_emission_cut

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
    pythia_process.readString("Print:quiet = on")  # Don't print anything but the basics
    pythia_process.readString("Print:init = off")  # Don't print all of the initialization business.
    pythia_process.readString("Print:next = off")  # Don't print all of the event business when we run a new event.

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
    if do_shower:
        # Enable parton-level interactions -- things will happen after the hard scattering.
        pythia_process.readString("PartonLevel:all = on")

        # We want the two hardest scattering outputs
        if not get_all:
            # Turn off multi-parton interactions
            pythia_process.readString("PartonLevel:MPI = off")

            # Turn off "Initial state" radiation with spacelike particles "before the hard process"
            pythia_process.readString("PartonLevel:ISR = off")

            # Turn off "Final state" radiation with timelike particles
            pythia_process.readString("PartonLevel:FSR = off")

            # Turn off beam remnants
            pythia_process.readString("PartonLevel:Remnants = off")  # Turn off beam remnant adding
            pythia_process.readString("Check:event = off") # Turn off event checks -- Missing beam remnants!

        # We want full jets, including beam remnants and such
        else:
            # Keep all of that!
            pass
    else:
        pythia_process.readString("PartonLevel:all = off")

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

        # Turn off heavy-flavor
        pythia_process.readString("HardQCD:gg2ccbar = off")
        pythia_process.readString("HardQCD:qqbar2ccbar = off")
        pythia_process.readString("HardQCD:hardccbar = off")
        pythia_process.readString("HardQCD:gg2bbbar = off")
        pythia_process.readString("HardQCD:qqbar2bbbar = off")
        pythia_process.readString("HardQCD:hardbbbar = off")

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

    # Here we bias the selection of pTHat for the process by a given power of pTHat (Here pTHat^4).
    # This is more or less equivalent to sampling from a uniform distribution in pTHat
    # and recording an appropriate true pTHat-dependent weight from a known weight distribution.
    pythia_process.readString("PhaseSpace:bias2Selection = on")
    pythia_process.readString("PhaseSpace:bias2SelectionPow = 4")

    # Set up to do a user veto and send it in.
    myUserHooks = MyUserHooks()
    pythia_process.setUserHooksPtr(myUserHooks)

    # Tell Pythia to "do the thing" (run with the configurations above)
    pythia_process.init()

    # Event loop. Iterate until getting a satisfactory hard process.
    success = False
    iEvent = -1
    while not success:
        iEvent += 1
        # Run the next event
        if not pythia_process.next(): continue
        nParticles = 0

        # Set if we're using the event record or the process record
        if pythia_process.event.size() == 0:
            num_particles_hist = pythia_process.process.size()
            record = pythia_process.process
        else:
            num_particles_hist = pythia_process.event.size()
            record = pythia_process.event

        # Confirm that our event only has particles we can handle
        allowed_partons = list(hard_particles._PARTICLE_SPECIES.keys())
        for i in np.arange(0, num_particles_hist):
            p = record[i]
            if (p.status() > 0):  # and p.isHadron() and p.isCharged():
                current_pid = p.id()
                if current_pid not in allowed_partons:
                    logging.debug("Disallowed particle: {}, Skipping event:{}".format(current_pid,
                                                                                         iEvent))
                    continue

        particle_list = []
        if get_all:
            # Collect all of the particles that exist in the final partonic state
            for i in np.arange(0, num_particles_hist):
                p = record[i]
                if (p.status() > 0):  # and p.isHadron() and p.isCharged():
                    nParticles += 1
                    particle_list = np.append(particle_list, p)
            success = True
            break
        else:
            # Iterate over all particles and look for the hardest partonic outputs.
            max_0 = 0
            max_0_i = 0
            max_1 = 0
            max_1_i = 0
            for i in np.arange(0, num_particles_hist):
                p = record[i]
                if (p.status() > 0):  # and p.isHadron() and p.isCharged():
                    nParticles += 1
                    if p.pT() > max_1:
                        if p.pT() > max_0:
                            max_0_i = i
                            max_0 = p.pT()
                        else:
                            max_1_i = i
                            max_1 = p.pT()
            ids = [record[max_0_i].id(), record[max_1_i].id()]
            ys = [record[max_0_i].y(), record[max_1_i].y()]
            particle_list = [record[max_0_i], record[max_1_i]]

            # Check if the results are The correct flavors, above pt cut, and in rapidity cut
            if type == "dijet":
                if ((np.abs(ids[0]) < 3.1) or (np.abs(ids[0]) == 21)) and ((np.abs(ids[1]) < 3.1) or (np.abs(ids[1]) == 21)):
                    if (max_0 > min_pt) and (max_1 > min_pt):
                        if np.abs(ys[0]) < y_res and np.abs(ys[1]) < y_res:
                            if ((max_0 - max_1)/max_0 < soft_emission_cut):
                                success = True
                                break  # Stop generating events, keep these particles

            if type == "gamma-jet":
                if (((np.abs(ids[0]) < 3.1) or (np.abs(ids[0]) == 21) or (np.abs(ids[0]) == 22))
                        and ((np.abs(ids[1]) < 3.1) or (np.abs(ids[1]) == 21) or (np.abs(ids[1]) == 22))):
                    if ids[0] == 22 or ids[1] == 22:  # At least one photon
                        if (max_0 > min_pt) and (max_1 > min_pt):
                            if np.abs(ys[0]) < y_res and np.abs(ys[1]) < y_res:
                                if ((max_0 - max_1)/max_0 < soft_emission_cut):
                                    success = True
                                    break  # Stop generating events, keep these particles

    ################################
    # Package and output particles #
    ################################
    logging.info("Event selection success: {}".format(success))
    # pythia_process.event.list()  # List the event that we accepted
    weight = pythia_process.infoPython().weight()
    output_particles = []

    # Shut down if we failed to produce any particles
    if len(particle_list) == 0:
        raise Exception("No particles found in event!")

    # Create a list of ape hard_particles.Particle objects
    for particle in particle_list:
        ape_particle = hard_particles.Particle.from_pythia(particle, tau=tau, x=x, y=y, etas=etas)
        output_particles.append(ape_particle)

    # Make an ape hard_particles.EventRecord object
    ape_event = hard_particles.EventRecord(particles=output_particles, weight=weight)

    return ape_event


# Function to take a list of particles and rotate the entire group by a random angle phi.
def phi_sample_embed(particles, weight):
    # Rotate the entire group by a random angle phi

    return particles, weight



# Function to hadronize a list of particles already including colors and anticolors and get pythia event
def pp_shower_hadronize(ape_event):
    logging.info('Hadronizing particles...')
    # Settings
    y_res = 1
    max_had_runs = 10000

    ############################################
    # Set up Pythia instance for hadronization #
    ############################################

    # Instantiate Pythia
    pythia_had = pythia8.Pythia("", False)  # Print header = False
    pythia_had.readString("Print:quiet = on")  # Don't print anything but the basics
    pythia_had.readString("Print:init = off")  # Don't print all of the initialization business.
    pythia_had.readString("Print:next = off")  # Don't print all of the event business when we hadronize.

    # Use seed based on time
    pythia_had.readString("Random:setSeed = on")
    pythia_had.readString("Random:seed = 0")

    # Only do the hadron level stuff
    pythia_had.readString("ProcessLevel:all = off")
    pythia_had.readString("PartonLevel:all = off")
    pythia_had.readString("HadronLevel:all = on")

    # Don't allow pi^0 to decay:
    # pythia_had.readString("111:mayDecay = off")

    # Allow color reconnection in hadronization
    # pythia_had.readString("ColourReconnection:forceHadronLevelCR = on")

    # Turn off event checks that enforce conservation of momentum and such in the event
    pythia_had.readString("Check:event = off")

    # Tell Pythia to "do the thing" (run with the configurations above)
    pythia_had.init()

    #################################
    # Run the hadronization routine #
    #################################
    # Try to hadronize until we get one that clears the event check.
    total_had_runs = 0
    while total_had_runs < max_had_runs:

        # Clear the event
        pythia_had.event.reset()

        # Add in edited particles -- note that this requires color connections already handled elsewhere.
        for p in ape_event.particles:
            scalein = 0.0 if p.scalein is None else float(p.scalein)

            pythia_had.event.append(
                id=int(p.id),
                status=int(23),  # typical: outgoing parton for hadronization input
                col=int(p.col),
                acol=int(p.acol),
                px=float(p.px),
                py=float(p.py),
                pz=float(p.pz),
                e=float(p.E0),     # uses on-shell energy with Pythia mass
                m=float(p.m0),     # uses Pythia's masses
                scaleIn=scalein,
            )

        # List particles for debug
        # pythia_had.event.list()

        # hadronize - restart if event checks fail
        event_success = pythia_had.next()
        if not event_success:
            total_had_runs += 1  # Add a total hadronization
            continue
        success_had_run = True
        total_had_runs += 1  # Add a total hadronization
        break

    # List particles again for debug
    logging.debug("Hadronization Event Check Success: {}".format(event_success))
    # pythia_had.event.list()

    logging.info('Hadronization complete.')
    return pythia_had.event

def pythia_to_fastjet(pythia_had: pythia8.Event, rap_max: float=1.5, R: float=0.4, pTmin: float=0.0):
    """
    Function to convert a pythia event to a group of fastjet pseudojets.

    Parameters
    ----------
    pythia_had : pythia8.Event to pull particles from
    rap_max : maximum absolute value of rapidity to consider
    R : Jet radius to consider
    pTmin : minimum pT to consider
    """
    # Collect particles fitting cuts as fastjet.PseudoJet objects
    particles = []
    for p in pythia_had.particles():
        # Filter
        if not p.isFinal(): continue  # Only consider final state particles
        if not p.isCharged(): continue  # Only consider charged particles
        if np.abs(p.y()) > rap_max: continue  # Rapidity cut
        if p.pT() < pTmin: continue  # Particle pT cut

        # Append
        particles.append(fastjet.PseudoJet(p.px(), p.py(), p.pz(), p.e()))  # px, py, pz, E

    # Jet algorithm definition
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    logging.info("Jetfinding using FastJet algorithm: {}".format(jet_def))

    # Find jets
    jets = jet_def(particles)

    return jets