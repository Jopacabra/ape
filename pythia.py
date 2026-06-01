import logging

import pythia8
import numpy as np
import fastjet
import pyhepmc

import config
import hard_particles

# Function to generate a pp hard scattering
def scattering(pThatmin=config.jet.pythia.PTHATMIN, pThatmax=config.jet.pythia.PTHATMAX, do_shower=config.jet.pythia.SHOWER,
               type=config.jet.pythia.TYPE, tau=config.soft_transport.all.TAU_FS, x=0, y=0, etas=0,
               selection_power=config.jet.pythia.BIAS_POWER,
               y_max=config.jet.pythia.RAP_MAX, y_min=config.jet.pythia.RAP_MIN, pythia_event=False, quiet=True,
               seed=0):
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

            # Get only events at mid-rapidity, within my chosen bounds
            abs_y = np.abs(info.y())
            if abs_y > y_max:  # and np.abs(chosen_pt -np.abs(info.pTHat())) < pt_hat_res:
                return True  # Veto the event
            elif abs_y < y_min:
                return True  # Veto the event
            else:
                logging.info(f"Hard scattering y = {info.y()}")
                return False  # Do not veto the event

    #################
    # Set up Pythia #
    #################
    pythia_process = pythia8.Pythia("", False)  # Print header = False
    if quiet:
        pythia_process.readString("Print:quiet = on")  # Don't print anything but the basics
        pythia_process.readString("Print:init = off")  # Don't print all of the initialization business.
        pythia_process.readString("Print:next = off")  # Don't print all of the event business when we run a new event.
    else:
        pass

    # Use seed based on time by default (0), or a specific seed as passed in the parameter
    pythia_process.readString("Random:setSeed = on")
    pythia_process.readString(f"Random:seed = {seed}")

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
    pythia_process.readString(f"PhaseSpace:bias2SelectionPow = {selection_power}")

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
        bad_particle = False
        allowed_partons = list(hard_particles._PARTICLE_SPECIES.keys())
        for i in np.arange(0, num_particles_hist):
            p = record[i]
            if (p.status() > 0):  # and p.isHadron() and p.isCharged():
                current_pid = p.id()
                if current_pid not in allowed_partons:
                    logging.debug("Disallowed particle: {}, Skipping event:{}".format(current_pid,
                                                                                         iEvent))
                    bad_particle = True
                    break

        if bad_particle:
            continue

        particle_list = []

        index_list = []
        # Collect all of the particles into a list
        for i in np.arange(0, num_particles_hist):
            p = record[i]
            # if (p.status() > 0):  # and p.isHadron() and p.isCharged():
            nParticles += 1
            particle_list = np.append(particle_list, p)
            index_list = np.append(index_list, int(i))
        success = True  # Breaks the while loop

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
    # num_particles = len(record.particles())
    for i, particle in enumerate(particle_list):
        ape_particle = hard_particles.Particle.from_pythia(particle, tau=tau, x=x, y=y, etas=etas, tag=i)
        output_particles.append(ape_particle)

    # Make an ape hard_particles.EventRecord object
    ape_event = hard_particles.EventRecord(particles=output_particles, weight=weight,
                                           event_tau0=tau, event_x0=x, event_y0=y, event_etas0=etas)

    if pythia_event:
        return ape_event, weight, record
    else:
        return ape_event, weight


# Function to take a list of particles and rotate the entire group by a random angle phi.
def phi_sample_embed(particles, weight):
    # Rotate the entire group by a random angle phi

    return particles, weight



# Function to hadronize a list of particles already including colors and anticolors and get pythia event
def ape_to_pythia(ape_event: hard_particles.EventRecord,
                  seed=0, hadronize=True):
    logging.info('Preparing final state Pythia event...')
    # Settings
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
    pythia_had.readString(f"Random:seed = {seed}")

    # Only do the hadron level stuff
    pythia_had.readString("ProcessLevel:all = off")
    pythia_had.readString("PartonLevel:all = off")
    pythia_had.readString("HadronLevel:all = on")

    # Don't allow pi^0 to decay:
    # pythia_had.readString("111:mayDecay = off")

    # Allow color reconnection in hadronization
    # pythia_had.readString("ColourReconnection:forceHadronLevelCR = on")

    # Event checks that enforce conservation of momentum and such in the event
    pythia_had.readString("Check:event = off")

    # Event checks that ensure mothers and daughters match
    pythia_had.readString("Check:history = off")

    # Tell Pythia to "do the thing" (run with the configurations above)
    pythia_had.init()

    #############################
    # Assemble the event record #
    #############################
    # Clear the event
    pythia_had.event.clear()  # Completely clears the event record, including the 0th "event as a whole" particle

    # Add in ape particles -- note that this requires color connections already handled elsewhere.
    # Includes 0th "event as a whole" particle
    for i, p in enumerate(ape_event.particles):  # Skip the first one that represents the event as a whole.
        scalein = 0.0 if p.scalein is None else float(p.scalein)
        pythia_had.event.append(
            id=int(p.id),
            status=int(p.status),  # typical: outgoing parton for hadronization input
            col=int(p.col),
            acol=int(p.acol),
            px=float(p.px),
            py=float(p.py),
            pz=float(p.pz),
            e=float(p.E0),  # uses on-shell energy with Pythia mass
            m=float(p.m0),  # uses Pythia's masses
            scaleIn=scalein,
            mother1=int(p.mother1),  # The particle's previous index, or the emitter's index
            mother2=int(p.mother2),
            daughter1=int(p.daughter1),
            daughter2=int(p.daughter2)
        )

    # Save the event for repeated hadronization
    saved_event = pythia_had.event

    #################################
    # Run the hadronization routine #
    #################################
    if hadronize:
        logging.info('Running Pythia string hadronization...')
        # Try to hadronize until we get one that clears the event check.
        total_had_runs = 0
        while total_had_runs < max_had_runs:

            # Reset to saved event state
            pythia_had.event = saved_event

            # List particles for debug
            # pythia_had.event.list()

            # hadronize - retry if remaining event checks fail
            event_success = pythia_had.next()
            if not event_success:
                total_had_runs += 1  # Add a total hadronization
                continue
            total_had_runs += 1  # Add a total hadronization
            break

        # List particles again for debug
        logging.debug("Hadronization Event Check Success: {}".format(event_success))
        # pythia_had.event.list()

        logging.info('Hadronization complete.')
    else:
        logging.info('Skipping Pythia string hadronization...')
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
        # if not p.isCharged(): continue  # Only consider charged particles
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


def pythia_to_hepmc(pythia_event: pythia8.Event, event_no=0, vx=None, vy=None, vz=None, vt=None, weight=1.0):
    """
    Function to convert a pythia event to a HepMC3 event, preserving parentage, using pyhepmc.

    This function manually creates vertices from mother-daughter relationships to avoid
    vertex counting bugs in pyhepmc.from_hepevt().

    Parameters
    ----------
    pythia_event : pythia8.Event
        The Pythia event record to convert
    event_no : int
        Event number for the HepMC event
    vx, vy, vz, vt : float, optional
        Vertex position (x, y, z, t) for all particles. If None, origin is used.

    Returns
    -------
    pyhepmc.GenEvent
        HepMC3 event with proper vertex structure
    """
    # Get a list of the particles
    particles = pythia_event.particles()

    # Create HepMC3 event
    hepmc_event = pyhepmc.GenEvent()
    hepmc_event.event_number = event_no

    # Set event info properties
    event_info_object = pyhepmc.GenEventData()
    event_info_object.event_pos = pyhepmc.FourVector(t=vt, x=vx, y=vy, z=vz)
    hepmc_event.read_data(event_info_object)

    # Set run info properties
    run_info_object = pyhepmc.GenRunInfo()
    run_info_object.weight_names = ["pythia"]

    # Attach to event object
    hepmc_event.run_info = run_info_object

    # Set weight
    hepmc_event.set_weight(name=str('pythia'), value=float(weight))

    # First pass: create all HepMC particles
    hepmc_particles = []
    for pythia_idx, p in enumerate(particles):

        # Skip the first element (represents event as a whole)
        if pythia_idx == 0: continue

        # Create HepMC particle
        hepmc_particle = pyhepmc.GenParticle()
        hepmc_particle.pid = p.id()
        hepmc_particle.momentum = pyhepmc.FourVector(p.px(), p.py(), p.pz(), p.e())
        hepmc_particle.status = p.statusHepMC()
        hepmc_particle.generated_mass = p.m0()

        # Append to list of particles for later access
        hepmc_particles.append(hepmc_particle)

    # Track which vertices have been created for each mother pair in a dictionary, with the keys as (mother1, mother2)
    # Each key corresponds to a vertex object
    mother_pair_to_vertex = {}

    # Second pass: build vertices from mother-daughter relationships
    for pythia_idx, p in enumerate(particles):
        mother1 = int(p.mother1())
        mother2 = int(p.mother2())

        # Normalize mother pair (always order them consistently)
        if mother1 == 0 and mother2 == 0:
            mother_pair = (0, 0)  # Primary vertex
        elif mother1 == 0:
            mother_pair = (0, mother2)
        elif mother2 == 0:
            mother_pair = (mother1, 0)
        else:
            mother_pair = (min(mother1, mother2), max(mother1, mother2))

        # Create vertex if it doesn't exist
        if mother_pair not in mother_pair_to_vertex:


            # Add mothers as incoming particles to this vertex
            if mother_pair == (0, 0):
                pass  # No vertex that generates the beam particles
            else:
                decay_vertex = pyhepmc.GenVertex()
                # decay_vertex.position = pyhepmc.FourVector(vx, vy, vz, vt)
                decay_vertex.status = 0 if mother_pair == (0, 0) else 1  # 0 for primary, 1 for decay
                hepmc_event.add_vertex(decay_vertex)
                mother_pair_to_vertex[mother_pair] = decay_vertex

                # Add actual mothers for decay vertices
                if mother1 > 0:
                    mother1_particle = hepmc_particles[mother1 - 1]  # bump one for Pythia indexing
                    decay_vertex.add_particle_in(mother1_particle)

                # Now treat other mothers according to cases
                # see Pythia docs (https://pythia.org//latest-manual/ParticleProperties.html):

                # Recoil effect or something
                if mother1 > 0 and mother1 == mother2:
                    pass  # Already handled above

                # A string fragmentation
                elif (mother2 > 0 and mother2 != mother1
                      and p.statusAbs() in [81, 82, 83, 84, 85, 86, 101, 102, 103, 104, 105, 106]):
                    for index in range(mother1 + 1, mother2 + 1):  # This is the start and stop index
                        mother2_particle = hepmc_particles[index - 1]  # bump one for Pythia indexing
                        decay_vertex.add_particle_in(mother2_particle)

                # Two truly different mothers
                elif mother2 > 0 and mother2 != mother1:
                    mother2_particle = hepmc_particles[mother2 - 1]  # bump one for Pythia indexing
                    decay_vertex.add_particle_in(mother2_particle)

        # Add this particle as outgoing from the vertex
        if mother_pair != (0, 0):
            decay_vertex = mother_pair_to_vertex[mother_pair]
            decay_vertex.add_particle_out(hepmc_particles[pythia_idx - 1])

    return hepmc_event


