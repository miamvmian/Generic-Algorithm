import numpy as np
import pandas as pd
from curve_fitting_GA.config import path

# wavelength to frequency (THz)
def lda2freq(lda):
    c0 = 3e8  # velocity of light
    return 2*np.pi*c0/(lda*1e-9)/1e12

def L_shape(omega, gamma, omega0, coef):
    return coef*(gamma/2)/((omega0-omega)**2+(gamma/2)**2)/np.pi

def eval_func(params):
    gamma, omega0, coef = params
    return L_shape(omega, gamma, omega0, coef)

def bin_len(low_param: float|int,
              upper_param: float|int,
              acc: float|int) -> int:
    num = int_bound(low_param,upper_param,acc)
    return len(format(num,'b'))

# convert real to binary
def int_bound(low_param: float|int,
               upper_param: float|int,
               acc: float|int) -> int:
    return int((upper_param-low_param)/acc)

# resonance wavelength(nm) and frequency(THz)
lda0=1442   # nanometer
omega00 = lda2freq(lda0)  # in THz

# import data to be fitted
path_TARGET = path/"data/R_part.csv"
DATA = pd.read_csv(filepath_or_buffer=path_TARGET, on_bad_lines='skip', index_col=None)
lda = DATA["0"]
TARGET = DATA["1"].values

# setting parameter ranges

gamma_range = (500, 1500) # gamma = Δω
omega0_range = (1000, 1500)   # eps_infinity range
coef_range = (500, 2e3)   # coef

# setting parameter accuracy
acc_gamma = 1e-1 # accuracy of gamma
acc_omega0 = 5e-1
acc_coef = 1e-1    # accuracy of coef

# calculate bit length for each parameter

bit_gamma_ = bin_len(gamma_range[0],gamma_range[1], acc_gamma)          # number of bits for gamma
bit_omega0_ = bin_len(omega0_range[0],omega0_range[1], acc_omega0)      # number of bits for omega0
bit_coef_ = bin_len(coef_range[0],coef_range[1], acc_coef)              # number of bits for coef

# calculate integer bound for each parameter
bound_gamma_ = int_bound(*gamma_range,acc_gamma)        # integer bound for gamma
bound_omega0_ = int_bound(*omega0_range,acc_omega0)     # integer bound for omega0
bound_omegaP_ = int_bound(*coef_range,acc_coef)     # integer bound for coef

# setting global constant
BITS = bit_gamma_, bit_omega0_, bit_coef_    # digits for parameters
BOUNDS = (bound_gamma_, bound_omega0_, bound_omegaP_)
omega = lda2freq(lda).round(1).values
PHYS_BOUNDS = (gamma_range, omega0_range, coef_range)


if __name__=="__main__":
    # ##### TEMP
    import matplotlib.pyplot as plt

    omega0 = 1.293e15
    gamma = 0.7693e15
    I0 = 0.8481e15
    c0 = 3e8

    def intensity(lda):
        return I0/np.pi*(gamma/2)/((omega0-2*np.pi/(lda*1e-9)*c0)**2+(gamma/2)**2)

    def phase(lda):
        return np.arctan(gamma/2/(2*np.pi/(lda*1e-9)*c0-omega0))

    plt.scatter(lda, TARGET, s=8)
    plt.plot(lda, intensity(lda), 'g-')
    ax = plt.gca()
    ax.set_facecolor('none')  # 坐标轴区域透明
    ax.set_facecolor('none')  # 画布背景透明

    # 设置坐标轴边框颜色为红色
    for spine in ax.spines.values():
        spine.set_color('red')
    # 设置刻度线和标签颜色为红色
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')

    # 设置坐标轴标签颜色为红色
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('red')
    plt.savefig(path/'transparent.png', transparent=True, dpi=600, bbox_inches='tight')
    plt.close()



    n_analyte = 1.03
    def phase_stretch(lda):
        return np.arctan(gamma/2/(2*np.pi*n_analyte*c0/(lda*1e-9)-omega0))

    lda1 = np.linspace(1000,2000,2001)
    phase1 = phase(lda1)
    phase2 =phase_stretch(lda1)

    neg1 = np.where(phase1<0)
    phase1[neg1] = phase1[neg1]+np.pi

    neg2 = np.where(phase2<0)
    phase2[neg2] = phase2[neg2]+np.pi

    plt.plot(lda1, phase1, 'k-', label=r"no analyte")
    plt.plot(lda1, phase2, '-.k', label=r"analyte")

    plt.legend()

    plt.savefig(path/'phase_stretch.png', transparent=True, dpi=600, bbox_inches='tight')
    plt.close()
