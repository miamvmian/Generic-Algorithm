import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import chain
import numpy as np
from typing import TypeVar, Iterable


PathLike = TypeVar("PathLike", str, Path)
ListLike = TypeVar('ListLike', list, tuple)

# FWHM for multiple peaks or valleys
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
        print("There is no peak.")
        return

    if len(arg):
        arg_start = arg[0]
        arg_end = arg[-1]
    else:
        print("Cannot find peaks.")
        return

    # calculate the partition index
    partition_indices = np.nonzero(np.diff(arg)-1)[0]
    partition_slices = []

    if len(partition_indices)==0:
        if arg_start == 0 and arg_end == len(data):
            print("No enough data for finding a peak.")
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


# search files ending with ".mph"
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


def search_CSVs(path: PathLike = Path.cwd()) -> Iterable|None:
    if path.exists():
        for full_path in path.iterdir():
            name = full_path.name
            if name.endswith('.csv'):
                yield full_path
    else:
        print(f"Cannot find the path: {path}")


if __name__ == "__main__":
    # File directory preparation
    folder = 'DataAnalysis/DataPostProcessing'
    sub_folder = 'Grating_GenericAlgorithm/analyte_refractive_index_variation'
    path = Path(folder)
    data_path = path/sub_folder

    data_path = path_regulator(data_path)
    CSVs300_strat = search_CSVs(data_path)

    # regex match
    import re
    # Define the regex pattern
    pattern300_R = r"COMSOL_.*_T=300_TM_R_thickness=\d+\[nm\]_slit=(\d+)\[nm\]_etch=(\d+)\[nm\]_n=(\d+\.\d+)"
    pattern300_A = r"COMSOL_.*_T=300_TM_A_thickness=\d+\[nm\]_slit=(\d+)\[nm\]_etch=(\d+)\[nm\]_n=(\d+\.\d+)"

    T300 = {"slit":[], "etch":[], "n":[],"c_wl":[],"c_val":[],"fwhm":[]}


    # group plot 1
    for fi in CSVs300_strat:
        fname = fi.stem
        # Match the pattern and extract the groups
        match300 = re.match(pattern300_R, fname)
        if match300:
            df = pd.read_csv(filepath_or_buffer=fi,
                             on_bad_lines='skip', index_col=0)
            df_clean = df.dropna(axis=1).copy()
            # df_clean = df_clean.drop([19])  # drop row number of 19 (irregular point)
            c_wl, c_val, fwhm = extract_FWHM_Wavelength(df_clean, [1.3e-6, 1.6e-6], peak=False)
            # Extract values from the capturing groups
            T300['slit'].append(match300.group(1).strip())
            T300['etch'].append(match300.group(2).strip())
            T300['n'].append(match300.group(3).strip())
            T300['c_wl'].append(c_wl)
            T300['c_val'].append(c_val)
            T300['fwhm'].append(fwhm)
        else:
            print("No match.")

    df300 = pd.DataFrame(T300)
    slit40 = df300[df300['slit'] == '40'].copy()
    # slit40['analyte'] = slit40['analyte'].astype(float)  # Modify the copy
    slit40 = slit40.sort_values(by='n', ascending=True, inplace=False)

    slit80 = df300[df300['slit'] == '80'].copy()
    slit80 = slit80.sort_values(by='n', ascending=True, inplace=False)

    slit80etch0 = slit80[slit80['etch']=='0'].copy()
    slit80etch30 = slit80[slit80['etch']=='30'].copy()

    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(slit40['n'], slit40['c_wl']*1e9, "--k.", label="over-etch 0[nm]")
    ax3.plot(slit80etch0['n'], slit80etch0['c_wl']*1e9, "-b*", label="over-etch 0[nm]")
    ax3.plot(slit80etch30['n'], slit80etch30['c_wl']*1e9, "-b^",label="over-etch 30[nm]")
    # hiding every two ticks showing in plot
    for label in ax3.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax3.set_ylim([1300, 1450])
    ax3.legend(loc='right', bbox_to_anchor=(1.0, 0.1))
    plt.savefig(data_path/"Resonace Wavelength.png", bbox_inches="tight", dpi=600)

    fig4, ax4 = plt.subplots(1, 1)
    ax4.plot(slit40['n'], slit40['fwhm']*1e9, "--k.", label="over-etch 0[nm]")
    ax4.plot(slit80etch0['n'], slit80etch0['fwhm']*1e9, "-b*", label="over-etch 0[nm]")
    ax4.plot(slit80etch30['n'], slit80etch30['fwhm']*1e9, "-b^",label="over-etch 30[nm]")
    # hiding every two ticks showing in plot
    for label in ax4.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax4.set_ylim([15, 80])
    ax4.legend(loc='right', bbox_to_anchor=(1.0, 0.3))
    plt.savefig(data_path/"FWHM.png", bbox_inches="tight", dpi=600)

    fig5, ax5 = plt.subplots(1, 1)
    ax5.plot(slit40['n'], slit40['c_val'], "--k.", label="over-etch 0[nm]")
    ax5.plot(slit80etch0['n'], slit80etch0['c_val'], "-b*", label="over-etch 0[nm]")
    ax5.plot(slit80etch30['n'], slit80etch30['c_val'], "-b^",label="over-etch 30[nm]")
    # hiding every two ticks showing in plot
    for label in ax5.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax5.set_ylim([0.15, 0.8])
    ax5.legend(loc='right', bbox_to_anchor=(1.0, 0.4))
    plt.savefig(data_path/"Reflectance at Resonance.png", bbox_inches="tight", dpi=600)
