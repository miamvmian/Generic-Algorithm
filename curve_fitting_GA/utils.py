import time
import numpy as np
import logging
import logging.handlers
from config import ListLike, Iterable, ListOrTuple, ListOrTuple2
from model import BITS, BOUNDS


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


# encoding parameters
def encode_parameters(params: float|int|Iterable[float|int],
                      bits: int|Iterable[int]=BITS) -> tuple:
    """
    Encodes the parameters into a single chromosome.
    Each parameter is encoded as a x-bit binary string.
    The final chromosome is the concatenation of the three genes,
    resulting in a X-bit binary string.
    """
    chromosome = []

    # convert to Iterable if not
    if not isinstance(bits, Iterable):
        params = (params,)
        bits = (bits,)

    for param, bit in zip(params, bits):
        param_bin = format(param, f'0{bit}b')  # encode parameter into binary with a given len
        gene = binary2Gray(param_bin)  # binary to Gray coding

        # Concatenate the genes to form the chromosome
        chromosome+=gene
    return tuple(chromosome)

# The decoding one chromosome into integer parameters
def decode_chromosome(chromosome: ListLike, bits: int|Iterable[int]=BITS)->int|Iterable:
    """
    Decodes a x-bit chromosome into the three parameters.

    The chromosome is assumed to be a string of 30 bits,
    where bits 0-9 represent h, bits 10-19 represent T, and bits 20-29 represent d.
    """
    if len(chromosome) != np.sum(bits):
        raise ValueError(f"Chromosome must be {len(chromosome)} bits long.")

    # Extract the segments
    num_bit = 0
    if isinstance(bits,int):
        num_bit += bits
        gene = chromosome[num_bit-bits:num_bit]
        binary = Gray2binary(gene)  # decode gray encoding
        return int(binary, 2)

    # if not Iterable
    int_params = []
    for bit in bits:
        num_bit += bit
        gene = chromosome[num_bit-bit:num_bit]
        binary = Gray2binary(gene)  # decode gray encoding
        int_params.append(int(binary, 2))
    return tuple(int_params)


# convert parameter (in form of integer) into real parameter
def int2phys(params_int: int|Iterable, bounds_int: int|Iterable,
              params_bounds: ListOrTuple|ListOrTuple2)->float|Iterable:

    if not isinstance(params_int, Iterable):
        param_low, param_high = params_bounds
        return param_low + params_int*(param_high-param_low)/bounds_int

    # if Iterable
    phys_params = []
    for param_int, bound_int, param_bounds in zip(params_int, bounds_int, params_bounds):
        param_low, param_high = param_bounds
        phys_params.append(param_low + param_int*(param_high-param_low)/bound_int)
    return tuple(phys_params)


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


def individual_generator(bounds: ListOrTuple=BOUNDS) -> Iterable:
    # Generate an individual
    ind = []
    for upper in bounds:
        ind.append(np.random.randint(0, upper))
    # encode parameters into chromosomes plus Gray encode
    return encode_parameters(ind, BITS)

# constraints for integer parameters
def param_constraints(params: int|Iterable[int],
                      bounds: int|Iterable[int]=BOUNDS)->bool:
    # params = integer version of (omegaP, gamma, epsInf)
    # bounds = integer version of boundary of (omegaP, gamma, epsInf)
    if isinstance(bounds, int):
        return params<bounds

    for param, bound in zip(params, bounds):
        if param>bound:
            return False
    return True

# mask function
def mask_func(params):
    time.sleep(3)
    return np.array(params)


# Relative Root Mean Squared Error
# y is the targeted value, y_hat is the estimated value
def RRMSE(y: ListLike, y_hat: ListLike) -> float:
    return np.sqrt(((np.array(y) - np.array(y_hat)) ** 2).mean()) / np.array(y).mean()

def RMSE(y: ListLike, y_hat: ListLike) -> float:
    return np.sqrt(((np.array(y) - np.array(y_hat)) ** 2).mean())

# Log-Cosh Loss
def LOG_COSH(y: ListLike, y_hat: ListLike) -> float:
    error = y - y_hat
    return np.mean(np.log(np.cosh(error)))