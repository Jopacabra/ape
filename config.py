import yaml
import os
import logging

############################################################
# See config.yml for descriptions of configuration options #
############################################################
# Get location of config.py and config.yml
project_path = os.path.dirname(os.path.realpath(__file__))

# Read config file and parse settings
try:
    with open('/srv/scratch/user_config.yml', 'r') as ymlfile:  # New location where htcondor will put input files
        logging.info('Using user edited config file')
        # Note the usage of yaml.safe_load()
        # Using yaml.load() exposes the system to running any Python commands in the config file.
        # That is unnecessary risk!!!
        cfg = yaml.safe_load(ymlfile)
except:
    try:
        with open(project_path + '/user_config.yml', 'r') as ymlfile:
            logging.info('Using user edited config file')
            # Note the usage of yaml.safe_load()
            # Using yaml.load() exposes the system to running any Python commands in the config file.
            # That is unnecessary risk!!!
            cfg = yaml.safe_load(ymlfile)
    except:
        with open(project_path + '/config.yml', 'r') as ymlfile:
            logging.info('Using default config file')
            # Note the usage of yaml.safe_load()
            # Using yaml.load() exposes the system to running any Python commands in the config file.
            # That is unnecessary risk!!!
            cfg = yaml.safe_load(ymlfile)


# Event by event sampling configuration
class EBE:
    NUM_EVENTS = int(cfg['mode']['NUM_EVENTS'])
    NUM_SAMPLES = int(cfg['mode']['NUM_SAMPLES'])
    NUM_FRAGS = int(cfg['mode']['NUM_FRAGS'])


# Mode configuration
class mode:
    VARY_POINT = bool(cfg['mode']['VARY_POINT'])
    KEEP_EVENT = bool(cfg['mode']['KEEP_EVENT'])
    KEEP_RECORD = bool(cfg['mode']['KEEP_RECORD'])


class transport:
    GRID_STEP = float(cfg['transport']['GRID_STEP'])
    TIME_STEP = float(cfg['transport']['TIME_STEP'])
    GRID_MAX_TARGET = float(cfg['transport']['GRID_MAX_TARGET'])

    class trento:
        NORM = float(cfg['trento']['NORM'])
        PROJ1 = str(cfg['trento']['PROJ1'])
        PROJ2 = str(cfg['trento']['PROJ2'])
        NUCLEON_WIDTH = float(cfg['trento']['NUCLEON_WIDTH'])
        CROSS_SECTION = float(cfg['trento']['CROSS_SECTION'])
        P = float(cfg['trento']['P'])
        K = float(cfg['trento']['K'])
        V = float(cfg['trento']['V'])
        NC = int(cfg['trento']['NC'])
        DMIN = float(cfg['trento']['DMIN'])

        try:
            BMIN = float(cfg['trento']['BMIN'])
        except ValueError:
            BMIN = None
        try:
            BMAX = float(cfg['trento']['BMAX'])
        except ValueError:
            BMAX = None

    class hydro:
        TAU_FS = float(cfg['transport']['TAU_FS'])
        T_SWITCH = float(cfg['transport']['T_SWITCH'])
        ETAS_MIN = float(cfg['transport']['ETAS_MIN'])
        ETAS_SLOPE = float(cfg['transport']['ETAS_SLOPE'])
        ETAS_CURV = float(cfg['transport']['ETAS_CURV'])
        ZETAS_MAX = float(cfg['transport']['ZETAS_MAX'])
        ZETAS_WIDTH = float(cfg['transport']['ZETAS_WIDTH'])
        ZETAS_T0 = float(cfg['transport']['ZETAS_T0'])


# Jet configuration
class jet:
    TAU_PROD = bool(cfg['jet']['TAU_PROD'])
    PTHATMIN = float(cfg['jet']['PTHATMIN'])
    PTHATMAX = float(cfg['jet']['PTHATMAX'])
    PROCESS_CORRECTIONS = bool(cfg['jet']['PROCESS_CORRECTIONS'])
    DTAU = float(cfg['jet']['DTAU'])
    T_HRG = float(cfg['jet']['T_HRG'])
    T_UNHYDRO = float(cfg['jet']['T_UNHYDRO'])
    K_F_DRIFT = float(cfg['jet']['K_F_DRIFT'])
    K_FG_DRIFT = float(cfg['jet']['K_FG_DRIFT'])
    K_BBMG = 1  #float(cfg['jet']['K_BBMG'])
    PT_BIN = float(cfg['jet']['PT_BIN'])
    TYPE = str(cfg['jet']['TYPE'])


# Global constants
class constants:
    G_RAD = float(cfg['global_constants']['G_RAD'])
    G_COL = float(cfg['global_constants']['G_COL'])
    G = G_RAD
    ROOT_S = float(cfg['global_constants']['ROOT_S'])
