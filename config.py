import yaml
import os
import logging

############################################################
# See config.yml for descriptions of configuration options #
############################################################
# Get location of config.py and config.yml
project_path = os.path.dirname(os.path.realpath(__file__))

class StopConfiguring(Exception):
    """ Raise to stop configuring """

# Read config file and parse settings
try:
    with open('/srv/scratch/user_config.yml', 'r') as ymlfile:  # New location where htcondor will put input files
        logging.info('Using user edited config file')
        # Note the usage of yaml.safe_load()
        # Using yaml.load() exposes the system to running any Python commands in the config file.
        cfg = yaml.safe_load(ymlfile)
except:
    try:
        with open(project_path + '/user_config.yml', 'r') as ymlfile:
            logging.info('Using user edited config file')
            # Note the usage of yaml.safe_load()
            # Using yaml.load() exposes the system to running any Python commands in the config file.
            cfg = yaml.safe_load(ymlfile)
    except:
        logging.error("No config file. Please provide user_config.yml.")
        print("No config file. Please provide user_config.yml.")

        raise StopConfiguring("No config file. Please provide user_config.yml. Ending run...")


# Mode configuration
class mode:
    try:
        SEED = int(cfg['mode']['SEED'])
    except:
        logging.info("No seed provided. Using random seed.")
        import random
        SEED = random.randint(0, 900000000)  # 900 million seeds -- pythia limitation

    VARY_POINT = bool(cfg['mode']['VARY_POINT'])
    KEEP_EVENT = bool(cfg['mode']['KEEP_EVENT'])
    WRITE_DATAFRAME = bool(cfg['mode']['WRITE_DATAFRAME'])
    EVENT_TYPE = str(cfg['mode']['EVENT_TYPE'])
    NUM_HARD = int(cfg['mode']['NUM_HARD'])


# Soft sector transport model configuration
class soft_transport:
    class all:
        GRID_STEP = float(cfg['soft_transport']['all']['GRID_STEP'])
        GRID_MAX = float(cfg['soft_transport']['all']['GRID_MAX'])
        TIME_STEP = float(cfg['soft_transport']['all']['TIME_STEP'])
        TAU_FS = float(cfg['soft_transport']['all']['TAU_FS'])

    class trento:
        NORM = float(cfg['soft_transport']['trento']['NORM'])
        PROJ1 = str(cfg['soft_transport']['trento']['PROJ1'])
        PROJ2 = str(cfg['soft_transport']['trento']['PROJ2'])
        NUCLEON_WIDTH = float(cfg['soft_transport']['trento']['NUCLEON_WIDTH'])
        CROSS_SECTION = float(cfg['soft_transport']['trento']['CROSS_SECTION'])
        P = float(cfg['soft_transport']['trento']['P'])
        K = float(cfg['soft_transport']['trento']['K'])
        V = float(cfg['soft_transport']['trento']['V'])
        NC = int(cfg['soft_transport']['trento']['NC'])
        DMIN = float(cfg['soft_transport']['trento']['DMIN'])

        try:
            BMIN = float(cfg['soft_transport']['trento']['BMIN'])
        except ValueError:
            BMIN = None
        try:
            BMAX = float(cfg['soft_transport']['trento']['BMAX'])
        except ValueError:
            BMAX = None

    class hydro:
        T_SWITCH = float(cfg['soft_transport']['hydro']['T_SWITCH'])
        ETAS_MIN = float(cfg['soft_transport']['hydro']['ETAS_MIN'])
        ETAS_SLOPE = float(cfg['soft_transport']['hydro']['ETAS_SLOPE'])
        ETAS_CURV = float(cfg['soft_transport']['hydro']['ETAS_CURV'])
        ZETAS_MAX = float(cfg['soft_transport']['hydro']['ZETAS_MAX'])
        ZETAS_WIDTH = float(cfg['soft_transport']['hydro']['ZETAS_WIDTH'])
        ZETAS_T0 = float(cfg['soft_transport']['hydro']['ZETAS_T0'])

    class frzout:
        MIN_SAMPLES = int(cfg['soft_transport']['frzout']['MIN_SAMPLES'])
        MAX_SAMPLES = int(cfg['soft_transport']['frzout']['MAX_SAMPLES'])
        MIN_PARTICLES = int(cfg['soft_transport']['frzout']['MIN_PARTICLES'])


# Jet configuration
class jet:
    TAU_PROD = float(cfg['jet']['TAU_PROD'])
    class pythia:
        TYPE = str(cfg['jet']['pythia']['TYPE'])
        PTHATMIN = float(cfg['jet']['pythia']['PTHATMIN'])
        PTHATMAX = float(cfg['jet']['pythia']['PTHATMAX'])
        BIAS_POWER = int(cfg['jet']['pythia']['BIAS_POWER'])
        SHOWER = bool(cfg['jet']['pythia']['SHOWER'])
        RAP_MIN = float(cfg['jet']['pythia']['RAP_MIN'])
        RAP_MAX = float(cfg['jet']['pythia']['RAP_MAX'])
    DTAU = float(cfg['jet']['DTAU'])
    T_HRG = float(cfg['jet']['T_HRG'])
    K_F_DRIFT = float(cfg['jet']['K_F_DRIFT'])
    K_BBMG = 1  #float(cfg['jet']['K_BBMG'])
    RAP_MAX_EVOLVE = float(cfg['jet']['RAP_MAX_EVOLVE'])
    EMIN = float(cfg['jet']['EMIN'])


# Global constants
class constants:
    G = float(cfg['global_constants']['G'])
    ROOT_S = float(cfg['global_constants']['ROOT_S'])
