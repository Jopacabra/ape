import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import logging
import collision
import plasma
import jets
import config
import utilities
from utilities import tempDir
import timekeeper
import pythia
import hadronization
import observables
import traceback
import argparse

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--eventtype", help="Type of initial conditions to use")

# Get command line arguments
args = parser.parse_args()
event_type = str(args.eventtype)  # Type of initial conditions to use

# Function to downcast datatypes to minimum memory size for each column
def downcast_numerics(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':  # for integers
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # for floats.
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Computes angular distance-ish quantity
def delta_R(phi1, phi2, y1, y2):
    dphi = np.abs(phi1 - phi2)
    if (dphi > np.pi):
        dphi = 2*np.pi - dphi
    drap = y1 - y2
    return np.sqrt(dphi * dphi + drap * drap)

# Exits temporary directory, saves dataframe to pickle, and dumps all temporary data.
def safe_exit(resultsDataFrame, event_obs, temp_dir, filename, identifier, hadrons_df=None, keep_event=False):
    # Save hydro event file
    if keep_event:
        logging.info('Saving event hydro data...')
        # Copy config file to results directory, tagged with identifier
        try:
            utilities.run_cmd(*['mv', 'viscous_14_moments_evo.dat',
                                results_path + '/hydro_grid_{}.dat'.format(identifierString, identifierString)],
                              quiet=False)
        except FileNotFoundError:
            logging.error('Failed to copy grid file -- file not found')

        try:
            utilities.run_cmd(*['mv', 'surface.dat',
                                results_path + '/hydro_surface_{}.dat'.format(identifierString, identifierString)],
                              quiet=False)
        except FileNotFoundError:
            logging.error('Failed to copy surface file -- file not found')

    try:
        logging.info('Saving event UrQMD observables...')
        np.save(results_path + '/{}_observables.npy'.format(filename), event_obs)
    except Exception as error:
        logging.info("Failed to copy UrQMD file: {}".format(type(error).__name__))  # An error occurred: NameError
        traceback.print_exc()

    # Save the dataframe into the identified results folder
    logging.info('Saving progress...')
    logging.info('Partons...')
    logging.debug(resultsDataFrame)
    logging.info('Converting datatypes to reasonable formats...')
    resultsDataFrame = resultsDataFrame.convert_dtypes()
    logging.info('Downcasting numeric values to minimum memory size...')
    resultsDataFrame = downcast_numerics(resultsDataFrame)

    logging.info('Writing pickles...')
    resultsDataFrame.to_pickle(results_path + '/{}.pickle'.format(filename))  # Save dataframe to pickle

    # Return to the directory in which we ran the script.
    os.chdir(home_path)

    # Clear everything in the temporary directory and delete it, thereby deleting all event files.
    logging.info('Cleaning temporary directory...')
    logging.debug('This dumps all of the event data!')
    try:
        temp_dir.cleanup()
    except TypeError:
        pass


# Function to generate a new HIC event and sample config.NUM_SAMPLES jets in it.
def run_event(eventNo):

    # Generate empty results frame
    event_partons = pd.DataFrame({})
    event_hadrons = pd.DataFrame({})

    ###############################
    # Generate new event geometry #
    ###############################

    logging.info('Generating new event...')

    # Run event generation using config setttings
    # Note that we need write permissions in the working directory
    event_dataframe, event_observables = collision.generate_event(working_dir=None, IC_type=event_type)
    rmax = event_dataframe.iloc[0]['rmax']

    # Record seed selected
    seed = event_dataframe.iloc[0]['seed']

    # Record number of participants
    npart = event_dataframe.iloc[0]['npart']

    # Record event psi_2
    psi_2 = event_dataframe.iloc[0]['psi_2']

    # Open the hydro file and create file object for manipulation.
    plasmaFilePath = 'viscous_14_moments_evo.dat'
    file = plasma.osu_hydro_file(file_path=plasmaFilePath, event_name='seed: {}'.format(seed))

    # Create event object
    # This asks the hydro file object to interpolate the relevant functions and pass them on to the plasma object.
    event = plasma.plasma_event(event=file, name=eventNo, rmax=rmax)

    # Compute temperature statistics for the hydro initial state (after freestreaming)
    maxTemp, minTemp, meanTemp, medianTemp, stdTemp = event.temp_stats()
    event_dataframe['Tmax_event'] = maxTemp
    event_dataframe['Tmin_event'] = minTemp
    event_dataframe['Tmean_event'] = meanTemp
    event_dataframe['Tmedian_event'] = medianTemp
    event_dataframe['Tstd_event'] = stdTemp

    # Create Cold Nuclear Matter effect interpolator
    CNM_interp = collision.CNM_RAA_interp()

    ##################
    # "Jet" Analysis #
    ##################
    # Select angular bin centers fixed at elliptic flow attractors and repulsors.
    # phi_res = np.pi/2
    # phi_bin_centers = np.arange(0, 2*np.pi, phi_res) + psi_2
    phi_rng = np.random.default_rng()
    num_phi = 11  # We select a prime number so this can't (?) influence v_n =/= v_{num_phi}

    # Oversample the background with jet seeds
    for process_num in range(0, config.EBE.NUM_SAMPLES):
        # Create unique jet seed tag
        process_tag = int(np.random.uniform(0, 1000000000000))
        logging.info('- Jet Seed Process {} Start -'.format(process_num))

        try:
            #########################
            # Create new scattering #
            #########################
            particles, weight = pythia.scattering(type=config.jet.TYPE)
            particle_tags = np.random.default_rng().uniform(0, 1000000000000, len(particles)).astype(int)
            process_partons = pd.DataFrame({})
            process_hadrons = pd.DataFrame({})

            # Compute angules for particles
            part_phis = np.array([])
            for index, particle in particles.iterrows():
                # Find angle of each particle, add to list
                pythia_phi = np.arctan2(particle['py'], particle['px']) + np.pi
                part_phis = np.append(part_phis, pythia_phi)

            # set coordinate system such that phi of first particle is at 0, on interval 0 to 2pi
            part_phis = np.mod(part_phis - part_phis[0], 2*np.pi)

            # Select jet seed production point
            if not config.mode.VARY_POINT:
                x0 = 0
                y0 = 0
            else:
                newPoint = collision.generate_jet_seed_point(event)
                x0, y0 = newPoint[0], newPoint[1]

            process_run = 0

            # Random azimuthal sampling
            # phi_values = phi_rng.uniform(0, 2 * np.pi, num_phi)

            # Uniform azimuthal sampling
            phi_values = np.linspace(start=0, stop=2*np.pi, num=num_phi, endpoint=False) #+ psi_2

            for phi_val in phi_values:
                # phi_val = np.mod(np.random.uniform(phi_center - phi_res/2, phi_center + phi_res/2), 2*np.pi)



                for case in [0, 1, 2, 3]:
                    case_partons = pd.DataFrame({})
                    # Determine case details
                    if case == 0:
                        el = True
                        cel = False
                        drift = False
                        fg = False
                        fgqhat = False
                        config.constants.G = config.constants.G_RAD
                    elif case == 1:
                        el = True
                        cel = False
                        drift = True
                        fg = False
                        fgqhat = False
                        config.constants.G = config.constants.G_RAD
                    elif case == 2:
                        el = True
                        cel = True
                        drift = False
                        fg = False
                        fgqhat = False
                        config.constants.G = config.constants.G_COL
                    elif case == 3:
                        el = True
                        cel = True
                        drift = True
                        fg = False
                        fgqhat = False
                        config.constants.G = config.constants.G_COL
                    else:
                        el = True
                        cel = False
                        drift = True
                        fg = False
                        fgqhat = False
                        config.constants.G = config.constants.G_RAD

                    if drift == True:
                        kfdrift_list = [1.0, 0.75, 1.25]
                    else:
                        kfdrift_list = [0.0]
                    for kfdrift in kfdrift_list:
                        config.jet.K_F_DRIFT = kfdrift
                        kf_partons = pd.DataFrame({})
                        i = 0
                        jet_seed_num = -1
                        for index, particle in particles.iterrows():
                            # Only do the things for the particle output
                            particle_status = particle['status']
                            particle_tag = int(particle_tags[i])
                            jet_seed_num += 1
                            # Read jet seed particle properties
                            chosen_e = particle['pt']
                            chosen_weight = weight
                            particle_pid = particle['id']
                            if particle_pid == 21:
                                chosen_pilot = 'g'
                            elif particle_pid == 1:
                                chosen_pilot = 'd'
                            elif particle_pid == -1:
                                chosen_pilot = 'dbar'
                            elif particle_pid == 2:
                                chosen_pilot = 'u'
                            elif particle_pid == -2:
                                chosen_pilot = 'ubar'
                            elif particle_pid == 3:
                                chosen_pilot = 's'
                            elif particle_pid == -3:
                                chosen_pilot = 'sbar'
                            elif particle_pid == 22:
                                chosen_pilot = 'gamma'
                                i += 1
                                continue  # Skip the photons -- noninteracting.
                            else:
                                i += 1
                                continue  # We don't know how this should interact... skip it.


                            # Select jet seed particle angles
                            phi_0 = np.mod(part_phis[i] + phi_val, 2*np.pi)

                            # Yell about your selected jet
                            logging.info('Pilot parton: {}, pT: {} GeV'.format(chosen_pilot, chosen_e))

                            el_model = 'num_GLV'

                            # Log jet number and case description
                            logging.info('Running Jet {}, case {}'.format(str(process_num), case))
                            logging.info('Energy Loss: {}, Vel Drift: {}, FG Drift: {}, FG Qhat: {}'.format(el, drift, fg,
                                                                                                            fgqhat))
                            # Perform AA CNM weighting
                            AA_weight = CNM_interp.weight(pt=chosen_e, npart=npart, id=particle_pid) * chosen_weight

                            # Create the jet object
                            parton = jets.parton(x_0=x0, y_0=y0, phi_0=phi_0, p_T0=chosen_e, tag=particle_tag, no=jet_seed_num, part=chosen_pilot,
                                              weight=chosen_weight, AA_weight=AA_weight)

                            # Perform pp-level fragmentation
                            pp_frag_z = hadronization.frag(parton, num=config.EBE.NUM_FRAGS)
                            pp_frag_z_22 = hadronization.frag(parton, num=config.EBE.NUM_FRAGS,
                                                              ff_name="JAM22-FF_hadron_nlo")

                            # Run the time loop
                            jet_dataframe, jet_xarray = timekeeper.evolve(event=event, parton=parton, drift=drift,
                                                                          el=el, cel=cel, fg=fg, fgqhat=fgqhat,
                                                                          el_model=el_model)

                            # Save the xarray trajectory file
                            # Note we are currently in a temp directory... Save record in directory above.
                            if config.mode.KEEP_RECORD:
                                jet_xarray.to_netcdf('../{}_record.nc'.format(process_tag))

                            # Add scattering process tag
                            jet_dataframe['process'] = process_tag

                            # Merge the event and jet dataframe lines
                            current_parton = pd.concat([jet_dataframe, event_dataframe], axis=1)

                            logging.info('FF Fragmentation')
                            # Perform ff fragmentation
                            frag_z = hadronization.frag(parton, num=config.EBE.NUM_FRAGS)
                            frag_z_22 = hadronization.frag(parton, num=config.EBE.NUM_FRAGS,
                                                           ff_name="JAM22-FF_hadron_nlo")
                            hadron_pt = parton.p_T() * frag_z[0]
                            hadron_pt_0 = parton.p_T0 * pp_frag_z[0]
                            hadron_22_pt = parton.p_T() * frag_z_22[0]
                            hadron_22_pt_0 = parton.p_T0 * pp_frag_z_22[0]
                            current_parton['z'] = [frag_z]
                            current_parton['pp_z'] = [pp_frag_z]
                            current_parton['z_22'] = [frag_z_22]
                            current_parton['pp_z_22'] = [pp_frag_z_22]
                            current_parton['hadron_pt_f'] = hadron_pt
                            current_parton['hadron_pt_0'] = hadron_pt_0
                            current_parton['hadron_22_pt_f'] = hadron_22_pt
                            current_parton['hadron_22_pt_0'] = hadron_22_pt_0
                            current_parton['process_run'] = process_run

                            # Save jet pair
                            if i == 0:
                                parton1 = parton
                            elif i == 1:
                                parton2 = parton

                            # Append current partons to the current process run list
                            kf_partons = pd.concat([kf_partons, current_parton], axis=0)

                            i += 1

                        logging.info('Computing process-level observables')
                        # Compute acoplanarity
                        if config.jet.TYPE == "dijet":
                            angles = kf_partons['phi_f'].to_numpy()
                            pts = kf_partons['pt_f'].to_numpy()
                            had_pts = kf_partons['hadron_pt_f'].to_numpy()
                            had_22_pts = kf_partons['hadron_22_pt_f'].to_numpy()
                            aco = np.abs(np.abs(np.mod(angles[0] - angles[1] + np.pi, 2 * np.pi) - np.pi))
                            kf_partons['partner_pt_f'] = np.flip(pts)
                            kf_partons['partner_hadron_pt_f'] = np.flip(had_pts)
                            kf_partons['partner_hadron_22_pt_f'] = np.flip(had_22_pts)
                            kf_partons['aco'] = np.full(2, aco)

                        # Append to the case partons
                        case_partons = pd.concat([case_partons, kf_partons], axis=0)

                        process_run += 1

                    logging.info('Appending case results to process results')
                    process_partons = pd.concat([process_partons, case_partons], axis=0)


        except Exception as error:
            logging.info("An error occurred: {}".format(type(error).__name__))  # An error occurred: NameError
            logging.info('- Jet Process Failed -')
            traceback.print_exc()

        event_partons = pd.concat([event_partons, process_partons], axis=0)

        # Declare jet complete
        logging.info('- Jet Process ' + str(process_num) + ' Complete -')

    # Compute event metadata for xarray datasets
    event_mult = event_dataframe['mult'][0]
    event_e2 = event_dataframe['e2'][0]
    event_Tmax = event.max_temp()
    flow_N = event_dataframe['urqmd_flow_N'][0]
    soft_psi_n = {}
    soft_v_n = {}
    for n in [2, 3, 4]:
        # Compute vn and psin soft
        soft_psi_n[n] = np.angle(event_dataframe['urqmd_re_q_{}'.format(n)][0]
                              + 1j * event_dataframe['urqmd_im_q_{}'.format(n)][0])
        soft_v_n[n] = np.abs(event_dataframe['urqmd_re_q_{}'.format(n)][0]
                          + 1j * event_dataframe['urqmd_im_q_{}'.format(n)][0]) / flow_N

    # Do coalescence & save xarray histograms for drift & no drift cases in all K_F_DRIFT options
    logging.info('Iterating through cases:')
    for drift_bool in [True, False]:
        for cel_bool in [False, True]:
            for KF_val in event_partons[(event_partons['drift'] == drift_bool)
                                        & (event_partons['cel'] == cel_bool)]['K_F_DRIFT'].value_counts().index:
                logging.info('K_F_DRIFT = {}, cel = {}'.format(KF_val, cel_bool))
                # Histogram partons into an xarray dataarray, using the v2 optimized bin number -- 157 bins
                xr_partons = utilities.xarray_ify(event_partons, pt_series='pt_f', phi_series='phi_f', pid_series='id',
                                                  weight_series='AA_weight', drift=drift_bool, cel=cel_bool,
                                                  NUM_PHI=157, K_F_DRIFT=KF_val)

                # Make fragmentation xarrays for JAM20
                xr_frag_hadrons_AA = utilities.xarray_ify_ff(event_partons, pt_series='pt_f', phi_series='phi_f', z_series='z',
                                                          weight_series='AA_weight', drift=drift_bool, cel=cel_bool,
                                                            NUM_PHI=157, K_F_DRIFT=KF_val)
                xr_frag_hadrons_pA = utilities.xarray_ify_ff(event_partons, pt_series='pt_0', phi_series='phi_0', z_series='pp_z',
                                                            weight_series='AA_weight', drift=drift_bool, cel=cel_bool,
                                                            NUM_PHI=157, K_F_DRIFT=KF_val)
                xr_frag_hadrons_pp = utilities.xarray_ify_ff(event_partons, pt_series='pt_0', phi_series='phi_0',
                                                            z_series='pp_z',
                                                            weight_series='weight', drift=drift_bool, cel=cel_bool,
                                                            NUM_PHI=157, K_F_DRIFT=KF_val)

                # Make fragmentation xarrays for JAM20
                xr_frag_hadrons_22_AA = utilities.xarray_ify_ff(event_partons, pt_series='pt_f', phi_series='phi_f',
                                                             z_series='z_22',
                                                             weight_series='AA_weight', drift=drift_bool, cel=cel_bool,
                                                             NUM_PHI=157, K_F_DRIFT=KF_val)
                xr_frag_hadrons_22_pA = utilities.xarray_ify_ff(event_partons, pt_series='pt_0', phi_series='phi_0',
                                                             z_series='pp_z_22',
                                                             weight_series='AA_weight', drift=drift_bool, cel=cel_bool,
                                                             NUM_PHI=157, K_F_DRIFT=KF_val)
                xr_frag_hadrons_22_pp = utilities.xarray_ify_ff(event_partons, pt_series='pt_0', phi_series='phi_0',
                                                             z_series='pp_z_22',
                                                             weight_series='weight', drift=drift_bool, cel=cel_bool,
                                                             NUM_PHI=157, K_F_DRIFT=KF_val)

                # Perform coalescence at T = 155 MeV
                if KF_val == 1.0 or KF_val == 0.0:
                    logging.info('Coalescing...')
                    xr_coal_hadrons = hadronization.coal_xarray(xr_partons, T=0.155, max_pt=20)
                    da_list = [xr_partons, xr_frag_hadrons_AA, xr_frag_hadrons_pA, xr_frag_hadrons_pp, xr_coal_hadrons,
                               xr_frag_hadrons_22_AA, xr_frag_hadrons_22_pA, xr_frag_hadrons_22_pp]
                else:
                    xr_coal_hadrons = None
                    da_list = [xr_partons, xr_frag_hadrons_AA, xr_frag_hadrons_pA, xr_frag_hadrons_pp,
                               xr_frag_hadrons_22_AA, xr_frag_hadrons_22_pA, xr_frag_hadrons_22_pp]

                # Assign event attributes
                for da in da_list:
                    for n in [2, 3, 4]:
                        da.attrs['psi_{}_soft'.format(n)] = soft_psi_n[n]
                        da.attrs['v_{}_soft'.format(n)] = soft_v_n[n]
                    da.attrs['mult'] = event_mult
                    da.attrs['Tmax'] = event_Tmax
                    da.attrs['e_2'] = event_e2
                    da.attrs['seed'] = seed
                    da.attrs['drift'] = drift_bool
                    da.attrs['cel'] = cel_bool
                    da.attrs['K_F_Drift'] = KF_val

                logging.info('Saving dataarrays...')
                # Save xarray dataarrays for JAM20
                xr_partons.to_netcdf(results_path + '/{}_AA_partons_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))
                xr_frag_hadrons_AA.to_netcdf(results_path + '/{}_AA_JAM20_hadrons_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))
                xr_frag_hadrons_pA.to_netcdf(results_path + '/{}_pA_JAM20_hadrons_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))
                xr_frag_hadrons_pp.to_netcdf(results_path + '/{}_pp_JAM20_hadrons_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))

                # Save xarray dataarrays for JAM22
                xr_frag_hadrons_22_AA.to_netcdf(results_path + '/{}_AA_JAM22_hadrons_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))
                xr_frag_hadrons_22_pA.to_netcdf(results_path + '/{}_pA_JAM22_hadrons_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))
                xr_frag_hadrons_22_pp.to_netcdf(results_path + '/{}_pp_JAM22_hadrons_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))

                # Save xarray dataarrays for coalescence
                if KF_val == 1.0 or KF_val == 0.0:
                    xr_coal_hadrons.to_netcdf(results_path + '/{}_AA_coal_hadrons_drift{}_cel{}_KFD{}.nc'.format(
                        identifierString, drift_bool, cel_bool, KF_val))

                logging.info('Computing and saving observables...')
                # Compute partonic vns
                part_vns = observables.compute_vns(xr_partons, n_list=np.array([2, 3, 4]))
                part_obs = part_vns
                part_obs.to_netcdf(results_path + '/{}_AA_partons_OBSERVABLES_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))

                # Compute JAM20 vns and raa
                frag_vns = observables.compute_vns(xr_frag_hadrons_AA, n_list=np.array([2, 3, 4]))
                frag_raa = observables.compute_raa(xr_frag_hadrons_AA, xr_frag_hadrons_pp)
                frag_obs = xr.merge([frag_vns, frag_raa])
                frag_obs.to_netcdf(results_path + '/{}_AA_JAM20_hadrons_OBSERVABLES_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))

                # Compute JAM22 vns and raa
                frag_22_vns = observables.compute_vns(xr_frag_hadrons_22_AA, n_list=np.array([2, 3, 4]))
                frag_22_raa = observables.compute_raa(xr_frag_hadrons_22_AA, xr_frag_hadrons_22_pp)
                frag_22_obs = xr.merge([frag_22_vns, frag_22_raa])
                frag_22_obs.to_netcdf(results_path + '/{}_AA_JAM22_hadrons_OBSERVABLES_drift{}_cel{}_KFD{}.nc'.format(
                    identifierString, drift_bool, cel_bool, KF_val))

                # Compute coalescence vns
                if KF_val == 1.0 or KF_val == 0.0:
                    coal_vns = observables.compute_vns(xr_coal_hadrons, n_list=np.array([2, 3, 4]))
                    coal_raa_20 = observables.compute_raa(xr_coal_hadrons, xr_frag_hadrons_pp)
                    coal_obs = xr.merge([coal_vns, coal_raa_20])
                    coal_obs.to_netcdf(results_path + '/{}_AA_coal_hadrons_OBSERVABLES_drift{}_cel{}_KFD{}.nc'.format(
                        identifierString, drift_bool, cel_bool, KF_val))

    return event_partons, event_hadrons, event_observables


################
# Main Program #
################

# Instantiate counters
part = 0
eventNo = 0

# Set up results frame and filename.
temp_dir = None  # Instantiates object for interrupt before temp_dir created.
results = pd.DataFrame({})
hadrons = pd.DataFrame({})
identifierString = str(int(np.random.uniform(0, 9999999999999)))
resultsFilename = 'results' + identifierString + 'p' + str(part)

# Set running location as current directory - whatever the pwd was when running the script
project_path = os.path.dirname(os.path.realpath(__file__))  # Gets directory the EBE.py script is located in
home_path = os.getcwd()  # Gets working directory when script was run - results directory will be placed here
results_path = home_path + '/results/{}'.format(identifierString)  # Absolute path of dir where results files will live

# Make results directory
os.makedirs(results_path, exist_ok=True)

# Force unbuffered output
sys.stdout.flush()
sys.stderr.flush()

# Check if os can write to the logging directory
if not os.access(results_path, os.W_OK):
    print(f"Cannot write to {results_path}")
    logging.error(f"Cannot write to {results_path}")
    # Set maximally loose permissions
    os.chmod(results_path, 0o777)  # Set directory permissions to 777

# Create log file & configure logging to be handled into the file AND stderr
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(results_path + '/log_{}.log'.format(identifierString))
    ]
)

# Add environment info to logs
logging.info(f"Job running in: {home_path}")
logging.info(f"Python version: {sys.version}")

# Copy config file to results directory, tagged with identifier
logging.info('Copying config.yml to results...')
try:
    with open(project_path + '/user_config.yml', 'r') as ymlfile:
        pass
    utilities.run_cmd(
        *['cp', project_path + '/user_config.yml', results_path + '/user_config_{}.yml'.format(identifierString)],
        quiet=True)
except:
    utilities.run_cmd(*['cp', project_path + '/config.yml', results_path + '/config_{}.yml'.format(identifierString)],
                  quiet=True)

# Run event loop
try:
    while config.EBE.NUM_EVENTS == 0 or eventNo < config.EBE.NUM_EVENTS:
        # Create and move to temporary directory
        temp_dir = tempDir(location=results_path)
        logging.info(os.getcwd())

        # Generate a new HIC event and sample config.NUM_SAMPLES jets in it
        # Append returned dataframe to current dataframe
        event_results, event_hadrons, event_observables = run_event(eventNo=int(identifierString))
        results = pd.concat([results, event_results], axis=0)

        # Exits directory, saves all current data, and dumps temporary files.
        safe_exit(resultsDataFrame=results, hadrons_df=hadrons, event_obs=event_observables, temp_dir=temp_dir,
                  filename=resultsFilename, identifier=identifierString,
                  keep_event=config.mode.KEEP_EVENT)

        if len(results) > 10000:
            part += 1
            resultsFilename = 'results' + identifierString + 'p' + str(part)
            results = pd.DataFrame({})
            hadronssFilename = 'results' + identifierString + 'p' + str(part)
            hadronss = pd.DataFrame({})

        eventNo += 1

except KeyboardInterrupt as error:
    logging.exception('Interrupted!!!: {}'.format(str(error)))
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT, event_obs=event_observables)

except collision.StopEvent as error:
    logging.exception('HIC event error: {}'.format(str(error)))
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT, event_obs=event_observables)

except MemoryError as error:
    logging.exception('Memory error: {}'.format(str(error)))
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT, event_obs=event_observables)

except BaseException as error:
    logging.exception('Unhandled error: {}'.format(str(error)))
    logging.info('Attempting to clean up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT, event_obs=event_observables)

logging.info('Results identifier: {}'.format(identifierString))
logging.info('Successful clean exit!')
logging.info('Please have an excellent day. :)')

# At the end of the script, ensure all logs are flushed
for handler in logging.root.handlers:
    handler.flush()
