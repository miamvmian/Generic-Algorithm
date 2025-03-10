import time
from pathlib import Path
from typing import TypeVar, Iterable
import numpy as np
import mph
import pandas as pd
import logging
import logging.handlers


PathLike = TypeVar("PathLike", str, Path)
ListLike = TypeVar('ListLike', list, tuple, Iterable)


# Configure root logger
def setup_logger(log_file='optimization.log'):
    log_format = "%(asctime)s | %(levelname)-8s | %(processName)-10s | %(message)s"
    formatter = logging.Formatter(log_format)

    # Rotating file handler (10 MB per file, max 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

setup_logger()
logger = logging.getLogger(__name__)

# path regulator
def path_regulator(path: PathLike = Path.cwd()) -> PathLike:
    work_path = Path.cwd()
    wp_parents = work_path.parents
    if not path.exists():
        for parent in wp_parents:
            child = work_path.relative_to(parent)
            if path.is_relative_to(child):
                path = parent/path
                break
    return path


# Path initialization
folder = Path("FEM/Grating_GenericAlgorithm")
subfolder = "Generic_Algorithm"
path_INI = path_regulator(folder/subfolder)
MODEL_NAME = path_INI / "GA_overetching_Grating_TM.mph"

# COMSOL simulation
def fem_simulation(params: np.ndarray, wl_span: ListLike=(1.3e-6, 1.55e-6)) -> ListLike:
    """
    :param params: [slit width, period, over-etching]
    :wl_span: wavelength sweep range, unit [m]
    :return: Tamm plasmon resonance wavelengths
    """

    # client initialization
    client = mph.start()
    client.clear()
    logger.info(f"Clear up mph client {client}")
    # loading COMSOL model
    py_model = client.load(MODEL_NAME)
    logger.info(f"Simulate the model: {MODEL_NAME.name}")
    model = py_model.java

    # #######################
    # initializing parameters: period, width, Zigzag thickness, analyte refractive index
    # Zigzag parameters
    t_Au = 60
    # set parameters
    model.param().set("t_Au", f"{t_Au}[nm]")
    logger.debug(f"Set thickness of Grating: t_Au={t_Au}[nm]")

    # list all available data tables
    tables = list(model.result().table().tags())
    # create table 1
    if tables and 'tbl1' in tables:
        model.result().table("tbl1").clearTableData()
    else:
        model.result().table().create("tbl1", "Table")
    # create table 2
    if tables and 'tbl2' in tables:
        model.result().table("tbl2").clearTableData()
    else:
        model.result().table().create("tbl2", "Table")

    # ######################
    # looping parameters
    outer_loop_params3 = (1.000, 1.520, 1.260)  # refractive indeces of analyte
    output = np.empty(len(outer_loop_params3), dtype=float)

    # unpacking parameters
    param0, param1, param2 = params

    # filling the Air with analytes
    h_slit = f"{param0}[nm]"
    model.param().set("h_slit", f"{h_slit}")
    logger.debug(f"Set width of slit: h_slit={h_slit}")

    T = f"{param1}[nm]"
    model.param().set("T", f"{T}")
    logger.debug(f"Set period of grating: T={T}")

    # setting parameters
    over_etch = f"{param2}[nm]"
    model.param().set("over_etch", f"{over_etch}")
    logger.debug(f"Set over_etch depth: over_etch={over_etch}")

    for ith, param3 in enumerate(outer_loop_params3):
        n_port1 = param3

        model.param().set("n_port1", f"{n_port1}")
        logger.debug(f"Set refractive index of analyte: n_analyte={n_port1}")

        model.component('comp1').geom('geom1').run('fin')
        model.component('comp1').mesh('mesh1').run()

        # Study setting
        wl_l, wl_h = wl_span[0]*1e6, wl_span[1]*1e6
        plist = f"range({wl_l}[um], {(wl_h-wl_l)/100}[um],{wl_h}[um])"
        # Access the parametric sweep feature (assuming it's already named "param")
        param_sw = model.study("std1").feature("param")  # No need to recreate if it exists
        # Update the range for lda0 (index 0) in the parametric sweep
        param_sw.setIndex("pname", "lda0", 0)
        param_sw.setIndex("punit", "m", 0)
        param_sw.setIndex("plistarr", f"{plist}", 0)
        param_sw.set("pdistrib", True)

        model.study("std1").setGenPlots(False)
        # running the model
        logger.info(f"Model \"{py_model.name()}.mph\" is running ...")

        model.study("std1").run()
        # global evaluation
        gev_tags = list(model.result().numerical().tags())
        ## create global evaluation
        if 'gev1' not in gev_tags:
            model.result().numerical().create("gev1", "EvalGlobal")
        model.result().numerical("gev1").set("data", "dset2")
        # model.result().numerical("gev1").setIndex("looplevelinput", "first", 0) # for batch sweep
        # model.result().numerical("gev1").set("innerinput", "first")  # for parameter sweep
        model.result().numerical("gev1").setIndex("expr", "ewfd.Rorder_0", 0)
        model.result().numerical("gev1").set("table", "tbl1")
        model.result().numerical("gev1").run()
        model.result().numerical("gev1").computeResult()
        model.result().numerical("gev1").setResult()

        # data processing
        tableData = pd.DataFrame(model.result().table("tbl1").getReal())
        df_clean = tableData.dropna(axis=1).copy()
        c_wl, c_val, fwhm = extract_FWHM_Wavelength(df_clean, wl_span, peak=False)

        if c_wl:    # resonance wavelength not None
            output[ith] = c_wl*1e9  # storage the resonance wavelength [nm]
        else:
            logger.info(f"No Tamm plasmon resonance found.")
            output = None
            break

        # Clear all solution data
        solutions = list(model.sol().tags())
        if solutions:
            for sol_i in solutions:
                model.sol(sol_i).clearSolutionData()

    return output


def extract_FWHM_Wavelength(dataframe: pd.DataFrame,
                            wl_span: ListLike =None,
                            peak: bool=True)->tuple[float|None, float|None, float|None]:
        fwhm = FWHM(dataframe, wl_span, peak)
        if fwhm:
            wl_left, wl_right, peak_wl, peak_val = fwhm
            if not peak:
                peak_val = 1 - peak_val
            return peak_wl, peak_val, wl_right-wl_left
        else:
            return None, None, None


# Algorithm for finding the FWHM from a peak or a valley
def FWHM(dataframe: pd.DataFrame, wl_span: ListLike =None,
         peak: bool=True)->tuple[float, float, float, float]|None:
    # wl_low is the lower bound
    # wl_up is the upper bound
    if wl_span and len(wl_span)==2:
        wl_span = sorted(wl_span)
        wl_low = wl_span[0]
        wl_up = wl_span[1]
    else:
        wl_low = None
        wl_up = None

    data = dataframe
    cols = dataframe.columns
    wl_colm = cols[0]   # get column label
    val_colm = cols[1]  # get column label
    if not peak:
        data[val_colm] = 1-data[val_colm]
    if wl_low:
        data = data[data[wl_colm]>wl_low]
    if wl_up:
        data = data[data[wl_colm]<wl_up]

    data.index = range(len(data))   # reset the data index
    arg_max = data[val_colm].argmax()
    peak = data.iloc[arg_max]
    peak_wl = peak[wl_colm] # get peak wavelength
    peak_val = peak[val_colm]   # get peak value
    arg = np.argwhere(np.sign(data[val_colm]-peak_val/2) >= 0).flatten()


    # Check if there is any peak
    if peak_val==data[val_colm].values[0] or peak_val==data[val_colm].values[-1]:
        logger.info("There is no peak.")
        return

    if len(arg):
        arg_start = arg[0]
        arg_end = arg[-1]
    else:
        logger.info("Cannot find peaks.")
        return

    # calculate the partition index
    partition_indices = np.nonzero(np.diff(arg)-1)[0]

    if len(partition_indices)==0:
        if arg_start == 0 and arg_end == len(data):
            logger.info("No enough data for finding a peak.")
            return
        # if not enough data points of left half of peak, then FWHM=2*(right half of FWHM)
        elif arg_start == 0:
            stop = arg_end
            wl_right = data[wl_colm][stop]
            wl_left = 2*peak_wl-wl_right
            return wl_left, wl_right, peak_wl, peak_val
        elif arg_end == data.index[-1]:
            start = arg_start
            wl_left = data[wl_colm][start]
            wl_right = 2*peak_wl - wl_left
            return wl_left, wl_right, peak_wl, peak_val
        else:
            start = arg_start
            stop = arg_end
            wl_left = data[wl_colm][start]
            wl_right = data[wl_colm][stop]
            return wl_left, wl_right, peak_wl, peak_val


# encoding three parameters h, T and d
def encode_parameters(h, T, d):
    """
    Encodes the three parameters into a single chromosome.
    Each parameter is encoded as a x-bit binary string.
    The final chromosome is the concatenation of the three genes,
    resulting in a X-bit binary string.
    """
    bin_h = format(h, '010b')  # encode h in 10 bits binary
    bin_T = format(T, '010b')  # encode T in 10 bits binary
    bin_d = format(d, '06b')  # encode d in 6 bits binary

    gene_h = binary2Gray(bin_h)  # binary to Gray coding
    gene_T = binary2Gray(bin_T)  # binary to Gray coding
    gene_d = binary2Gray(bin_d)  # binary to Gray coding

    # Concatenate the genes to form the chromosome
    chromosome = gene_h + gene_T + gene_d
    return chromosome

# The corresponding decoding function
def decode_chromosome(chromosome):
    """
    Decodes a x-bit chromosome into the three parameters.

    The chromosome is assumed to be a string of 30 bits,
    where bits 0-9 represent h, bits 10-19 represent T, and bits 20-29 represent d.
    """
    if len(chromosome) != 26:
        raise ValueError(f"Chromosome must be {len(chromosome)} bits long.")

    # Extract the three 10-bit segments
    gene_h = chromosome[0:10]
    gene_T = chromosome[10:20]
    gene_d = chromosome[20:26]

    # Convert Gray code into binary
    bin_h =  Gray2binary(gene_h)  # Gray code to binary
    bin_T =  Gray2binary(gene_T)  # Gray dode to binary
    bin_d = Gray2binary(gene_d)  # Gray dode to binary

    # Convert binary strings back to integers
    h = int(bin_h, 2)   # decode h
    T = int(bin_T, 2)   # decode T
    d = int(bin_d, 2)   # decode d

    return h, T, d


def binary2Gray(binary_code):
    """
    Convert a binary string to its Gray code representation.
    """
    binary_str = ''.join(str(_) for _ in binary_code)
    # Convert binary string to integer
    binary_num = int(binary_str, 2)

    # Compute Gray code using XOR between binary_num and its right shift
    gray_num = binary_num ^ (binary_num >> 1)
    # Convert back to binary string with same length as input
    gray_str = format(gray_num, f'0{len(binary_str)}b')
    return [int(s) for s in gray_str]


def Gray2binary(gray_code):
    """
    Convert a Gray code string to its binary representation.
    """
    gray_str = ''.join(str(_) for _ in gray_code)
    binary_str = gray_str[0]  # MSB remains the same

    # Iterate through remaining bits
    for i in range(1, len(gray_str)):
        # XOR current Gray bit with previous binary bit
        binary_bit = str(int(gray_str[i]) ^ int(binary_str[i - 1]))
        binary_str += binary_bit

    return binary_str


def individual_generator():
    # Generate an individual
    while True:
        h = np.random.randint(41, 51)
        T = np.random.randint(300, 601)
        d = np.random.randint(0, 31)

        # constraints, makes Tamm plasmon resonance wavelength
        # not far from 1.4um
        # if 200 <= T-h <= 300:
        if T and h and d:
            break
    # encode parameters into chromosomes/individuals
    return encode_parameters(h, T, d)

# constraints to parameter h, T and d
def param_constraints(params):
    h, T, d = params
    # return 80<=h<=299 and 300<=T<=600 and 0<=d<=30 and 150<T-h<350
    return 40<h<51 and 300<=T<=600 and 0<=d<=30

# mask function
def mask_func(params):
    time.sleep(3)
    return np.array(params)

# Relative Root Mean Squared Error
# y is the targeted value, y_hat is the estimated value
def RRMSE(y: ListLike, y_hat: ListLike) -> float:
    return np.sqrt(((np.array(y) - np.array(y_hat)) ** 2).mean()) / np.array(y).mean()
