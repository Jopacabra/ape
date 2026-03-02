# Anisotropic Parton Evolution (APE)
Anisotropic parton evolution in hydrodynamic QGP grids generated with the DukeQCD hic-eventgen package (https://github.com/Duke-QCD/hic-eventgen).

### Dependencies
#### External Python modules required:

* scipy
* numpy
* cython
* h5py
* pandas
* xarray
* py-yaml
* hic
* tkinter (For plasma inspector only)
* matplotlib (For plotting only)
* jupyter (For sample scripts only)

#### Trento & OSU-Hydro dependencies

* Python 3.5+ with numpy, scipy, cython, and h5py
* C, C++, and Fortran compilers
* CMake 3.4+
* Boost and HDF5 C++ libraries

#### Requires LHAPDF 

* Installed separately, e.g. as in the included install.sh script
* Requires Python interface

#### Requires Pythia

* Installed separately, e.g. as in the included install.sh script
* Requires Python interface

### Usage

APE was built specifically for deployment on the Open Science Grid (OSG) via container distribution. Included is a container build script, an event running script, and a submission script in the "osg" folder. The workflow goes as follows:

1. Build the container with apptainer:
    e.g.: apptainer build container.sif image.def
2. Update the submission script with the location of the container image on the OSPool.
3. Submit script. Make sure to edit config.yml, rename to user_config.yml, and pass as input file with submission script. No default configuration is provided.
4. Collect result "*.pickle" files containing parton results

### Data & Analysis

APE outputs results in pickle files containing a Pandas dataframe. 

Each row of the dataframe corresponds to a single parton's initial and final properties. (It is possible to configure
the suite to output xarray dataarrays with the step-by-step properties of the parton saved to netCDF, as is done by the 
tk_plasma_inspector.py script, by specifying (KEEP_RECORD: TRUE) in the config.yml. The output is labelled 
appropriately.) Partons are tagged with an id specifying the event in which they were evolved and some details about 
the event. Most of the properties are not used in the analysis from the companion paper. 

Generally, it is convenient to write a script to combine many of these pickle files into a single dataframe, then to 
compute desired event-averaged observables. Sample scripts used in <> will be included in "sample_scripts".

The full list of properties is as follows 
(<x> suggests replaceable variable for repeated entries):

'partonNo' - The identifier if the parton is from hard parton output 0 or 1 in the hard scattering. Identifies particle 
          within a hard scattering event.

'tag' - Unique particle identifier to globally identify multiple cases (different effects on or off) of the same 
        initial conditions

'weight' - Pythia-assigned weight from importance sampling of the scattering pT-hat distribution. Partons were sampled 
           from a rescaled distribution so as to collect information on different regions in pT. This roughly 
           corresponds to something like the "number of times this hard scattering occurs proportionate to original pT-hat
           distribution." This should be considered the weight in pp collisions.

'AA_weight' - pp weight x reweighting factor for cold nuclear matter effects in AA collisions.

'id' - PDG id of the parton "https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf"

'pt_0' - pT of the parton directly after the hard scattering

'pt_f' - pT of the parton after interacting with the QGP (if any effects turned on)

'q_el' - Net energy change due to radiative energy loss (usually a negative number corresponding to energy loss)

'q_cel' - Net energy change due to collisional energy loss (usually a negative number corresponding to energy loss)

'q_drift' - Net transverse momentum change due to flow-mediated jet drift (+ corresponds to counter-clockwise turn) 

'q_drift_abs' - Sum of absolute value of transverse momentum change due to flow-mediated jet drift in each step
                -- total drift momentum transfer, not net transfer (+ corresponds to ccw turn) 

'q_fg_<x>' - Net transverse momentum change due to flow-(gradient of <x>)-mediated jet drift 
              (+ corresponds to counter-clockwise turn) 

'q_fg_<x>_abs' - Sum of absolute value of transverse momentum change due to flow-(gradient of <x>)-mediated jet drift 
                 in each step -- total drift momentum transfer, not net transfer (+ corresponds to ccw turn)

'extinguished' - Legacy tag -- suggests the parton lost more energy than it had. Should be considered unphysical.

'x_0' - x position of the initial hard scattering

'y_0' - y position of the initial hard scattering

'phi_0' - Initial azimuthal angle of trajectory directly after hard scattering

'phi_f' - Final azimuthal angle of trajectory after QGP interaction (if any)

't_qgp' - Event time at which the parton first sees a QGP -- Should correspond to the freestreaming time.

't_hrg' - Event time at which the parton first sees temperatures below hadronization temp. (set in config.yml)

't_unhydro' - Event time at which the parton first sees temperatures below unhydrodynamic temp. (set in config.yml)

'time_total_plasma' - Total event time / ~ pathlength parton sees in a QGP

'time_total_hrg' - Total event time / ~ pathlength parton sees between hadronization temp. and unhydrodynamic temp.

'time_total_unhydro'- Total event time / ~ pathlength parton sees below unhydrodynamic temp.

'Tmax_parton' - Maximum temperature seen by the parton

'Tavg_qgp_parton' - ~pathlength/proper time - averaged temperature seen by parton

'initial_time' - Time at which parton propagation / hydrodynamics begins. Should correspond to the free-streaming time.

'final_time' - Final timestep of the hydrodynamics

'dtau' - timestep size used for parton propagation

'Tmax_event' - Maximum temperature of the event the parton was evolved in (parton did not necessarily see this Temp.)

'K_F_DRIFT' - Multiplicative factor on the strength of drift in this trajectory.

'drift' - Boolean flag determining if flow-mediated drift was enabled.

'el' - Boolean flag determining if radiative energy loss was enabled.

'cel' - Boolean flag determining if collisional energy loss was enabled.

'el_num' - Boolean flag determining if radiative energy loss was computed numerically or with analytic approx.

'fg' - Boolean flag determining if flow-gradient mediated drift was enabled (unstudied and untested!!!)

'fgqhat' - Boolean - if flow-gradient modification of qhat was enabled for rad. energy loss (unstudied and untested!!!)

'exit' - Reason for terminating parton propagation:
         0. Parton escaped event geometry (spatial grid)
         1. Parton lost more energy in a single step than it had -- Something went wrong
         2. Parton escaped event time, but had exited plasma 
            (temporal hydro grid bounds -- event temp enforced below config T_SWITCH)
         3. Parton escaped event time, but had NOT exited plasma -- Something went wrong

'g' - In-medium coupling constant used by this parton -- set by config.yml

'process' - Unique identifier of the hard scattering event used to seed this particle. Should match for "dijet" pairs

'b' - Impact parameter of the event this parton was evolved in

'npart' - (Trento) Number of participants of the event this parton was evolved in

'ncoll' - (Trento) Number of binary collisions of the event this parton was evolved in

'mult' - (Trento) Multiplicity of the event this parton was evolved in

'e<n>_re' - Real part of complex <n>th eccentricity vector for the event this parton was evolved in 
            (e.g. <n>=2 for elliptic vector)

'e<n>_im' - Imaginary part of complex <n>th eccentricity vector of (Trento) initial conditions for the event this parton 
            was evolved in (e.g. <n>=2 for elliptic vector)

'psi_e<n>' - Angle of the <n>th eccentricity vector of (Trento) initial conditions for the event this parton was evolved
             in (e.g. <n>=2 for elliptic vector)

'seed' - Random seed supplied to Trento to generate this event. Event should be deterministic up to Cooper-Frye sampling

'rmax' - Maximum event size from center of grid used to measure required hydro grid size -- See DukeQCD's hic-eventgen

'e2' - Mag. of the <n>th eccentricity vector of (Trento) initial conditions for the event this parton was evolved in
       (e.g. <n>=2 for elliptic vector)

'urqmd_re_q_<n>' - real part of <n>th Flow Q-vectors computed from UrQMD outputs

'urqmd_im_q_<n>' - imaginary part of <n>th Flow Q-vectors computed from UrQMD outputs

'urqmd_flow_N' - Total number of particles for flow sum as output by UrQMD (oversampled -- This is not the multiplicity)

'urqmd_dNch_deta' - Total number of charged particles differential in pseudo-rapidity from UrQMD

'initial_entropy' - Initial entropy of Trento initial conditions

'urqmd_nsamples' - Number of Cooper-Frye samples used when feeding particles to UrQMD

'urqmd_dN_dy_<species>' - Number of particles differential in rapidity for particle <species> 

'urqmd_mean_pT_<species>' - Mean pT of particles for particle <species>

'urqmd_pT_fluct_N' - pT fluctuations -- see DukeQCD hic-eventgen

'urqmd_pT_fluct_sum_pT' - pT fluctuations -- see DukeQCD hic-eventgen

'urqmd_pT_fluct_sum_pTsq' - pT fluctuations -- see DukeQCD hic-eventgen

'urqmd_dET_deta' - pT fluctuations -- see DukeQCD hic-eventgen

'psi_2' - Elliptic soft particle (from UrQMD) event plane angle

'v_2' - Elliptic soft particle (from UrQMD) flow

'z' - Sampled Fragmentation momentum fractions as sampled using 'pt_f'

'pp_z' - Sampled Fragmentation momentum fractions as sampled using 'pt_0'

'hadron_pt_f' - 'z' * 'pt_f' -- Momentum of possible hard hadron with QGP interactions

'hadron_pt_0' - 'pp_z' * 'pt_0' -- Momentum of possible hard hadron without QGP interactions

'process_run' - Iteration of this hard process -- Each hard process is run for the inclusion of energy loss 
                or energy loss and drift

'aco' - Acoplanarity (smallest angle between particles) from this dihadron event

