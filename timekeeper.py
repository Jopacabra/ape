import numpy as np
import pandas as pd
import logging
import plasma_interaction as pi
import config
import xarray as xr
from scipy import interpolate
import os
import traceback
import utilities


def evolve(event, parton, drift=True, el=True, fg=True, fgqhat=False, cel=False, scale_drift=1, scale_el=1, el_model='GLV',
           temp_hrg=config.jet.T_HRG, temp_unh=config.jet.T_UNHYDRO):
    parton_dataframe = pd.DataFrame({})  # Empty dataframe to return in case of issue.
    # If using numerical energy loss, summon the interpolator
    if el_model == 'num_GLV':
        el_rate_interp = pi.num_eloss_interpolator()
        el_num = True
    else:
        el_num = False

    #############
    # Time Loop #
    #############
    # Set loop parameters
    dtau = config.jet.DTAU  # dt for time loop in fm
    tau = event.t0  # Set current time in fm to initial time

    # Initialize counters & values
    t_qgp = -1
    t_hrg = -1
    t_unhydro = -1
    qgp_time_total = 0
    hrg_time_total = 0
    unhydro_time_total = 0
    maxT = 0
    q_el_total = 0
    q_cel_total = 0
    q_drift_total = 0
    q_drift_abs_total = 0
    q_fg_utau_total = 0
    q_fg_utau_abs_total = 0
    q_fg_uperp_total = 0
    q_fg_uperp_abs_total = 0
    q_fg_utau_qhat_total = 0
    q_fg_utau_qhat_abs_total = 0
    q_fg_uperp_qhat_total = 0
    q_fg_uperp_qhat_abs_total = 0

    # Initialize flags
    first = True
    qgp_first = True
    hrg_first = True
    unhydro_first = True
    phase = None
    extinguished = False

    # Initialize jet info storage arrays
    time_array = np.array([])
    xpos_array = np.array([])
    ypos_array = np.array([])
    q_drift_array = np.array([])
    q_el_array = np.array([])
    q_cel_array = np.array([])
    q_fg_utau_array = np.array([])
    q_fg_uperp_array = np.array([])
    q_fg_utau_qhat_array = np.array([])
    q_fg_uperp_qhat_array = np.array([])
    pT_array = np.array([])
    temp_seen_array = np.array([])
    grad_perp_T_seen_array = np.array([])
    grad_perp_utau_seen_array = np.array([])
    grad_perp_uperp_seen_array = np.array([])
    u_perp_array = np.array([])
    u_par_array = np.array([])
    u_array = np.array([])
    phase_array = np.array([])

    # Set failsafe values
    rho_final = 0
    phi_final = 0
    pT_final = 0

    # If we want to produce hard scattering at t=0, we should propagate the partons
    # in the time before thermalization...
    parton.prop(tau=(tau - config.jet.TAU_PROD))


    # Initiate loop
    logging.info('Initiating time loop...')
    while True:
        #logging.info('t = {}'.format(t))
        #####################
        # Initial Step Only #
        #####################
        if first:
            # Good luck!
            first = False

        #########################
        # Set Current Step Data #
        #########################
        # Decide if we're in bounds of the grid
        if parton.x > event.xmax or parton.y > event.ymax or parton.x < event.xmin or parton.y < event.ymin:
            logging.info('Parton escaped event space...')
            exit_code = 0
            break
        elif tau > event.tf:
            logging.info('Parton escaped event time...')
            if phase == 'qgp':
                exit_code = 3
            else:
                exit_code = 2
            break

        # Record p_T at beginning of step for extinction check
        parton_og_p_T = parton.p_T()

        # For timekeeping in phases, we approximate all time in one step as in one phase
        parton_point = parton.coords3(time=tau)
        parton_p_rho, parton_p_phi = parton.polar_mom_coords()
        beta = parton.beta()

        temp = utilities.dtau_avg(func=event.temp, point=parton_point, phi=parton_p_phi, dtau=config.jet.DTAU, beta=beta) #event.temp(parton_point)
        grad_perp_T = utilities.dtau_avg(func=lambda x: event.grad_perp_T(point=x, phi=parton_p_phi),
                                          point=parton_point, phi=parton_p_phi, dtau=config.jet.DTAU, beta=beta)  #event.grad_perp_T(point=parton_point, phi=parton_p_phi)
        grad_perp_utau = utilities.dtau_avg(func=lambda x: event.grad_perp_u_par(point=x, phi=parton_p_phi),
                                          point=parton_point, phi=parton_p_phi, dtau=config.jet.DTAU, beta=beta)  #event.grad_perp_u_par(point=parton_point, phi=parton_p_phi)
        grad_perp_uperp = utilities.dtau_avg(func=lambda x: event.grad_perp_u_perp(point=x, phi=parton_p_phi),
                                          point=parton_point, phi=parton_p_phi, dtau=config.jet.DTAU, beta=beta)  #event.grad_perp_u_perp(point=parton_point, phi=parton_p_phi)
        u_perp = event.u_perp(point=parton_point, phi=parton_p_phi)
        u_par = utilities.dtau_avg(func=lambda x: event.u_par(point=x, phi=parton_p_phi), point=parton_point, phi=parton_p_phi,
                               dtau=config.jet.DTAU, beta=beta)
        u = utilities.dtau_avg(func=event.vel, point=parton_point, phi=parton_p_phi, dtau=config.jet.DTAU, beta=beta)  #event.vel(parton_point)

        # Decide phase
        if temp > temp_hrg:
            phase = 'qgp'
        elif temp < temp_hrg and temp > temp_unh:
            phase = 'hrg'
        elif temp < temp_unh and temp > config.transport.hydro.T_SWITCH:
            phase = 'unh'
        else:
            phase = 'vac'

        #################################
        # Perform partonic calculations #
        #################################

        if phase == 'qgp':
            # Compute drift, if enabled
            if drift:
                # Compute jet drift integrand in this timestep
                int_drift = pi.drift_integrand(event=event, parton=parton, time=tau)

                # Compute jet drift momentum transferred to parton
                q_drift = float(parton.beta() * dtau * int_drift * scale_drift)
            else:
                # Set drift integral and momentum transfer to zero
                int_drift = 0
                q_drift = 0

            # Compute energy loss, if enabled
            if el:
                # Use appropriate energy loss module
                # Compute energy loss integrand (rate) in this timestep
                if el_model == 'num_GLV':
                    int_el = el_rate_interp.eloss_rate(event=event, parton=parton, time=tau)

                else:
                    # Compute energy loss integrand (rate) in this timestep
                    int_el = pi.energy_loss_integrand(event=event, parton=parton, time=tau, tau=dtau,
                                                      model=el_model)


                # Compute energy loss due to gluon exchange with the medium
                q_el = float(parton.beta() * dtau * int_el * scale_el)
            else:
                # Set energy loss and el integral to zero
                int_el = 0
                q_el = 0

            if cel:
                # Compute energy loss integrand (rate) in this timestep
                int_cel = pi.coll_energy_loss_integrand(event=event, parton=parton, time=tau)
                # Compute energy loss due to gluon exchange with the medium
                q_cel = float(parton.beta() * dtau * int_cel)
            else:
                int_cel = 0
                q_cel = 0

            if fg:
                # Compute mixed flow-gradient drift integrand in this timestep
                int_fg_utau = pi.flowgrad_utau_integrand(event=event, parton=parton, time=tau)
                int_fg_uperp = pi.flowgrad_uperp_integrand(event=event, parton=parton, time=tau)

                # Compute momentums transferred to parton
                q_fg_utau = float(parton.beta() * dtau * int_fg_utau)
                q_fg_uperp = float(parton.beta() * dtau * int_fg_uperp)
            else:
                # Set flow-gradient effects and integral to zero
                int_fg_utau = 0
                int_fg_uperp = 0
                q_fg_utau = 0
                q_fg_uperp = 0

            if fgqhat:
                # Compute correction to energy loss due to flow-gradient modification
                int_fg_utau_qhat = int_el * pi.fg_utau_qhat_mod_factor(event=event, parton=parton, time=tau)
                int_fg_uperp_qhat = int_el * pi.fg_uperp_qhat_mod_factor(event=event, parton=parton, time=tau)
                q_fg_utau_qhat = float(parton.beta() * dtau * int_fg_utau_qhat * scale_el)
                q_fg_uperp_qhat = float(parton.beta() * dtau * int_fg_uperp_qhat * scale_el)
            else:
                # Set correction to energy loss due to flow-gradient modification to zero
                int_fg_utau_qhat = 0
                int_fg_uperp_qhat = 0
                q_fg_utau_qhat = 0
                q_fg_uperp_qhat = 0

        else:
            # If not in QGP, don't compute any parton-medium interactions
            # If you wanted to add some effects in other phases, they should be computed here
            int_el = 0
            q_el = 0

            int_cel = 0
            q_cel = 0

            int_drift = 0
            q_drift = 0

            int_fg_utau = 0
            int_fg_uperp = 0
            q_fg_utau = 0
            q_fg_uperp = 0

            int_fg_utau_qhat = 0
            int_fg_uperp_qhat = 0
            q_fg_utau_qhat = 0
            q_fg_uperp_qhat = 0

        ###################
        # Data Accounting #
        ###################
        # Log momentum transfers
        q_el_total += q_el
        q_cel_total += q_cel
        q_drift_total += q_drift
        q_drift_abs_total += np.abs(q_drift)
        #q_fg_T_total += q_fg_T
        #q_fg_T_abs_total += np.abs(q_fg_T)
        q_fg_utau_total += q_fg_utau
        q_fg_utau_abs_total += np.abs(q_fg_utau)
        q_fg_uperp_total += q_fg_uperp
        q_fg_uperp_abs_total += np.abs(q_fg_uperp)
        q_fg_utau_qhat_total += q_fg_utau_qhat
        q_fg_utau_qhat_abs_total += np.abs(q_fg_utau_qhat)
        q_fg_uperp_qhat_total += q_fg_uperp_qhat
        q_fg_uperp_qhat_abs_total += np.abs(q_fg_uperp_qhat)

        # Check for max temperature
        if temp > maxT:
            maxT = temp  #[0]

        # Decide phase for categorization & timekeeping
        if phase == 'qgp':
            if qgp_first:
                t_qgp = tau
                qgp_first = False

            qgp_time_total += dtau

        # Decide phase for categorization & timekeeping
        if phase == 'hrg':
            if hrg_first:
                t_hrg = tau
                hrg_first = False

            hrg_time_total += dtau

        if phase == 'unh':
            if unhydro_first:
                t_unhydro = tau
                unhydro_first = False

            unhydro_time_total += dtau

        # Record arrays of values from this step for the parton record
        time_array = np.append(time_array, tau)
        xpos_array = np.append(xpos_array, parton.x)
        ypos_array = np.append(ypos_array, parton.y)
        q_drift_array = np.append(q_drift_array, q_drift)
        q_el_array = np.append(q_el_array, q_el)
        q_cel_array = np.append(q_cel_array, q_cel)
        #q_fg_T_array = np.append(q_fg_T_array, q_fg_T)
        q_fg_utau_array = np.append(q_fg_utau_array, q_fg_utau)
        q_fg_uperp_array = np.append(q_fg_uperp_array, q_fg_uperp)
        q_fg_utau_qhat_array = np.append(q_fg_utau_qhat_array, q_fg_utau_qhat)
        q_fg_uperp_qhat_array = np.append(q_fg_uperp_qhat_array, q_fg_uperp_qhat)
        pT_array = np.append(pT_array, parton.p_T())
        temp_seen_array = np.append(temp_seen_array, temp)
        grad_perp_T_seen_array = np.append(grad_perp_T_seen_array, grad_perp_T)
        grad_perp_utau_seen_array = np.append(grad_perp_utau_seen_array, grad_perp_utau)
        grad_perp_uperp_seen_array = np.append(grad_perp_uperp_seen_array, grad_perp_uperp)
        u_perp_array = np.append(u_perp_array, u_perp)
        u_par_array = np.append(u_par_array, u_par)
        u_array = np.append(u_array, u)
        phase_array = np.append(phase_array, phase)

        ############################
        # Change Parton Parameters #
        ############################
        # Note -- We propagate FIRST in order to travel over the timestep whose medium properties we're averaging.
        # Propagate parton position
        parton.prop(tau=dtau)

        # Change parton momentum to reflect energy loss
        parton.add_q_par(q_par=q_el)
        parton.add_q_par(q_par=q_cel)
        parton.add_q_par(q_par=q_fg_utau_qhat)
        parton.add_q_par(q_par=q_fg_uperp_qhat)

        # Change parton momentum to reflect drift effects
        # If not computed, q values go to zero.
        parton.add_q_perp(q_perp=q_drift)
        #parton.add_q_perp(q_perp=q_fg_T)
        parton.add_q_perp(q_perp=q_fg_utau)
        parton.add_q_perp(q_perp=q_fg_uperp)

        # Check if the "jet" would be extinguished (prevents flipping directions
        # when T >> p_T, since q_el has no p_T dependence):
        # If the parton lost more energy this step than it had
        # at the beginning of the step, we extinguish the "jet" and end things
        if np.abs(q_el) >= parton_og_p_T:
            logging.info('Parton extinguished')
            parton.p_x = 0
            parton.p_y = 0
            extinguished = True
            exit_code = 1
            break

        ###############
        # Timekeeping #
        ###############
        tau += dtau

        # Get final parton parameters
        rho_final, phi_final = parton.polar_mom_coords()
        pT_final = parton.p_T()

    logging.info('Time loop complete...')

    mean_QGP_temp = np.mean(temp_seen_array[phase_array == 'qgp'])
    # Create momentPlasma results dataframe
    try:
        print('Making dataframe...')
        parton_dataframe = pd.DataFrame(
            {
                "partonNo": [int(parton.no)],
                "tag": [int(parton.tag)],
                "weight": [float(parton.weight)],
                "AA_weight": [float(parton.AA_weight)],
                "id": [int(parton.id)],
                "pt_0": [float(parton.p_T0)],
                "pt_f": [float(pT_final)],
                "q_el": [float(q_el_total)],
                "q_cel": [float(q_cel_total)],
                "q_drift": [float(q_drift_total)],
                "q_drift_abs": [float(q_drift_abs_total)],
                "q_fg_utau": [float(q_fg_utau_total)],
                "q_fg_utau_abs": [float(q_fg_utau_abs_total)],
                "q_fg_uperp": [float(q_fg_uperp_total)],
                "q_fg_uperp_abs": [float(q_fg_uperp_abs_total)],
                "q_fg_utau_qhat": [float(q_fg_utau_qhat_total)],
                "q_fg_utau_qhat_abs": [float(q_fg_utau_qhat_abs_total)],
                "q_fg_uperp_qhat": [float(q_fg_uperp_qhat_total)],
                "q_fg_uperp_qhat_abs": [float(q_fg_uperp_qhat_abs_total)],
                "extinguished": [bool(extinguished)],
                "x_0": [float(parton.x_0)],
                "y_0": [float(parton.y_0)],
                "phi_0": [float(parton.phi_0)],
                "phi_f": [float(phi_final)],
                "t_qgp": [float(t_qgp)],
                "t_hrg": [float(t_hrg)],
                "t_unhydro": [float(t_unhydro)],
                "time_total_plasma": [float(qgp_time_total)],
                "time_total_hrg": [float(hrg_time_total)],
                "time_total_unhydro": [float(unhydro_time_total)],
                "Tmax_parton": [float(maxT)],
                "Tavg_qgp_parton": [float(mean_QGP_temp)],
                "initial_time": [float(event.t0)],
                "final_time": [float(event.tf)],
                "dtau": [float(config.jet.DTAU)],
                "drift": [bool(drift)],
                "el": [bool(el)],
                "cel": [bool(cel)],
                "el_num": [bool(el_num)],
                "fg": [bool(fg)],
                "fgqhat": [bool(fgqhat)],
                "exit": [int(exit_code)],
                "g": [float(config.constants.G)],
                "K_F_DRIFT": [float(config.jet.K_F_DRIFT)],
                "K_FG_DRIFT": [float(config.jet.K_FG_DRIFT)]
            }
        )

        logging.info('Pandas dataframe generated...')

    except Exception as error:
        logging.info("An error occurred: {}".format(type(error).__name__))  # An error occurred: NameError
        logging.info('- Parton Dataframe Creation Failed -')
        traceback.print_exc()

    # Create and store parton record xarray
    # define data with variable attributes
    logging.info('Creating xarray parton record...')
    data_vars = {'x': (['time'], xpos_array,
                       {'units': 'fm',
                        'long_name': 'x position coordinate'}),
                 'y': (['time'], ypos_array,
                       {'units': 'fm',
                        'long_name': 'y position coordinate'}),
                 'q_drift': (['time'], q_drift_array,
                             {'units': 'GeV',
                              'long_name': 'Momentum obtained by the parton at this timestep due to flow drift'}),
                 'q_fg_utau': (['time'], q_fg_utau_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the parton at this timestep due to flow-grad_utau drift'}),
                 'q_fg_uperp': (['time'], q_fg_uperp_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the parton at this timestep due to flow-grad_uperp drift'}),
                 'q_el': (['time'], q_el_array,
                            {'units': 'GeV',
                             'long_name': 'Momentum obtained by the parton at this timestep due to radiative energy loss'}),
                 'q_cel': (['time'], q_cel_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the parton at this timestep due to collisional energy loss'}),
                 'q_fg_utau_qhat': (['time'], q_fg_utau_qhat_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the parton at this timestep due to fg_utau mod to energy loss'}),
                 'q_fg_uperp_qhat': (['time'], q_fg_uperp_qhat_array,
                              {'units': 'GeV',
                               'long_name': 'Momentum obtained by the parton at this timestep due to fg_uperp mod to energy loss'}),
                 'pT': (['time'], pT_array,
                          {'units': 'GeV',
                           'long_name': 'Transverse momentum of the parton at this timestep'}),
                 'temp': (['time'], temp_seen_array,
                          {'units': 'GeV',
                           'long_name': 'Temperature seen by the parton at this timestep'}),
                 'grad_perp_temp': (['time'], grad_perp_T_seen_array,
                          {'units': 'GeV/fm',
                           'long_name': 'Gradient of Temperature perp. to parton seen by the parton at this timestep'}),
                 'grad_perp_utau': (['time'], grad_perp_utau_seen_array,
                                    {'units': 'GeV/fm',
                                     'long_name': 'Gradient of utau perp. to parton seen by the parton at this timestep'}),
                 'grad_perp_uperp': (['time'], grad_perp_uperp_seen_array,
                                    {'units': 'GeV/fm',
                                     'long_name': 'Gradient of uperp perp. to parton seen by the parton at this timestep'}),
                 'u_perp': (['time'], u_perp_array,
                            {'units': 'GeV',
                             'long_name': 'Temperature seen by the parton at this timestep'}),
                 'u_par': (['time'], u_par_array,
                           {'units': 'GeV',
                            'long_name': 'Temperature seen by the parton at this timestep'}),
                 'u': (['time'], u_array,
                       {'units': 'GeV',
                        'long_name': 'Temperature seen by the parton at this timestep'}),
                 'phase': (['time'], phase_array,
                           {'units': 'qgp = Quark Gluon Plasma, hrg = HadRon Gas, unh = UNHydrodynamic hadron gas, vac = below unh cutoff / vacuum',
                            'long_name': 'Phase seen by the parton at this timestep'})
                 }

    # define coordinates
    coords = {'time': (['time'], time_array)}

    # define global attributes
    attrs = {'property_name': 'value'}

    # create dataset
    parton_xarray = xr.Dataset(data_vars=data_vars,
                            coords=coords,
                            attrs=attrs)

    parton.record = parton_xarray

    logging.info('Xarray dataframe generated...')

    return parton_dataframe, parton_xarray
