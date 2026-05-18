"""
Converted from notebook: CalVal_postprocess_interactive.ipynb

This script contains code cells and markdown (as comments) extracted
from the original interactive notebook for easier scripting and
automation. Review and adapt paths or magic commands as needed.
"""

# ---------------------------------------------------------------------
#%% Postprocess SWOT CalVal Data
#
# This script continues the CalVal workflow after the download step.
# ---------------------------------------------------------------------

# Loads CalVal passes `9` and `22` for the Mid-Atlantic Bight (MAB)
# calibration/validation phase
# Processes cycles in the range 478-579
import sys
from pathlib import Path
sys.path.append("/home/sryan/python/")  # go to parent dir
from utils.plot_utils import finished_plot, latlon_label
from utils.datafun import datestr2doy, datetime2numeric, numeric2datetime
from utils.vector_fun import rotate, compute_angle
from utils.general import pickle_save, pickle_load

import xarray as xr
import numpy as np
from dask.distributed import Client

# load SWOT diagnoctis
# add the directory *above* the SwotDiag/ folder
sys.path.append("/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/SwotDiag")
from SwotDiag.diagnosis import *

# load project startup file with all relevant functions
exec(open('/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/startup_swot.py').read())

# path for saving calval plots
plotpath = './projects/NASA_SWOT_NWA_shelf_2024/analysis/plots/CalVal/Antarctica/'


#-----------------------------------------
#%% helper functions
#-----------------------------------------

def extract_cycle(files):
    import re
    cycle = []
    for file in files:
        m = re.search(r"Expert_(\d{3})", str(file))
        if not m:
            raise ValueError("Filename does not match expected pattern")
        cycle.append(int(m.groups()[0]))
    return cycle





# ----------------------------------------
#%% Merge and process pass data
# ----------------------------------------


def process_pass(pass_id, filepath, cycle_range=None, lat_bounds=(30, 55)):
    """Load and preprocess a single pass by `pass_id`.

    Returns an xarray Dataset with derived ADT, speed, standardized
    cycle/time coordinates, and latitude trimming.
    """
    directory = Path(filepath)
    matches = sorted(directory.glob(f"*Expert_???_{pass_id:03d}_*.nc"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No files found for pass {pass_id} in {filepath}")

    datasets = []
    for f in matches:
        dummy = xr.open_dataset(f)
        if 'i_num_pixel' in dummy:
            dummy = dummy.drop_vars(['i_num_pixel', 'i_num_line'])
        datasets.append(dummy)

    ds = xr.concat(datasets, dim='cycle')

    # derive ADT and speed
    ds['adt_filtered'] = ds['mdt'] + ds['ssha_filtered']
    ds['adt_unfiltered'] = ds['mdt'] + ds['ssha_unfiltered']
    ds['speed_filtered'] = np.sqrt(ds['ugos_filtered']**2 + ds['vgos_filtered']**2)

    # cycle numbers from filenames
    cycles = extract_cycle(matches)
    ds['cycle'] = (('cycle'), cycles)

    # numeric time for interpolation, reindex cycles to requested range
    if cycle_range is None:
        cycle_range = np.arange(474, 578)
    ds['timenum'] = (('cycle', 'num_lines'), datetime2numeric(ds.time, unit='s'))
    ds = ds.reindex(cycle=cycle_range)
    ds['time'] = (('cycle', 'num_lines'), numeric2datetime(ds.timenum, unit='s'))
    ds = ds.assign_coords(time_cycle=("cycle", ds.time.mean(dim='num_lines').values))
    ds = ds.swap_dims({'cycle': 'time_cycle'})

    # limit latitude bounds to reduce data size
    latmin, latmax = lat_bounds
    ds = ds.where((ds.latitude > latmin) & (ds.latitude < latmax), drop=True)

    return ds



# ------------------------------------------------------------------------
# ccall function
# ------------------------------------------------------------------------
filepath = '/srv/data/SWOT/L3/CalVal/v3_0/'
lat_bounds = (30, 55)
pass9 = process_pass(9,filepath=filepath, lat_bounds=lat_bounds)
pass22 = process_pass(22,filepath=filepath, lat_bounds=lat_bounds)




# -------------------------------------------------                                                                  
# %% Compute diagnostics
# -------------------------------------------------

def swot_add_diag(ds, n=9):
    """Compute relative vorticity and cyclogeostrophic velocities from filtered and unfiltered ADT."""
    params = dict(
        derivative='fit',
        n=n,
        min_valid_points=0.75,
        cyclostrophy='GW',
        avoid_negative=False,
        second_derivative='dxdy',
        kernel='circular',
    )

    dummy_unfilt = compute_ocean_diagnostics_from_eta(ds.adt_unfiltered, ds.longitude, ds.latitude, **params) # from diag package
    dummy_filt = compute_ocean_diagnostics_from_eta(ds.adt_filtered, ds.longitude, ds.latitude, **params)

    ds = ds.assign(
        zeta_filtered=(ds.ugos_filtered.dims, dummy_filt['zeta']),
        zeta_unfiltered=(ds.ugos_filtered.dims, dummy_unfilt['zeta']),
        ucg_filtered=(ds.ugos_filtered.dims, dummy_filt['ucg']),
        ucg_unfiltered=(ds.ugos_filtered.dims, dummy_unfilt['ucg']),
        vcg_filtered=(ds.ugos_filtered.dims, dummy_filt['vcg']),
        vcg_unfiltered=(ds.ugos_filtered.dims, dummy_unfilt['vcg'])
    )

    del dummy_unfilt, dummy_filt
    return ds


# -----------------------------------------------------
# Start local Dask cluster (this is set-up to run on a big machine so be aware of your resources)
# -----------------------------------------------------
client = Client(
    n_workers=4,
    threads_per_worker=6,
    memory_limit='20GB',
    dashboard_address=':8787'
) 

print(client)


test = swot_add_diag(pass9)




client.close()

























#%% OLD code for computing diagnostics using the diag package.
# Load previously saved diagnostics
pass9_diag = np.load('/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/derived_data/pass9_diag_calval.nc.npy', allow_pickle=True)
pass22_diag = np.load('/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/derived_data/pass22_diag_calval.nc.npy', allow_pickle=True)


def swot_add_diagnostics(ds, n=5):
    params = dict(derivative='fit', n=n, min_valid_points=0.75,
                  cyclostrophy='GW', avoid_negative=False,
                  second_derivative='dxdy', kernel='circular')

    diag = []
    for i in range(len(ds.time_cycle)):
        diag.append(compute_ocean_diagnostics_from_eta(ds.adt_unfiltered.isel(time_cycle=i),
                                                       ds.longitude, ds.latitude, **params))

    diag_dic = {}
    for var in diag[0].keys():
        dummy = []
        for i in range(len(ds.time_cycle)):
            dummy.append(diag[i][var])
        diag_dic[var] = dummy

    dims = ds.ugos_filtered.dims
    ds = ds.assign(ugos=(dims, diag_dic['ug']), vgos=(dims, diag_dic['vg']),
                   ucgos=(dims, diag_dic['ucg']), vcgos=(dims, diag_dic['vcg']),
                   zeta=(dims, diag_dic['zeta']), sr=(dims, diag_dic['S']),
                   OW=(dims, diag_dic['OW']))
    return ds[['ugos', 'vgos', 'ucgos', 'vcgos', 'zeta', 'sr', 'OW']]


pass9_diag = {}
pass22_diag = {}
for nfilt in [5, 9, 13]:
    pass9_diag[f'{nfilt}p'] = swot_add_diagnostics(pass9, n=nfilt)
    pass22_diag[f'{nfilt}p'] = swot_add_diagnostics(pass22, n=nfilt)
    pass9_diag[f'{nfilt}p']['Ug'] = np.sqrt(pass9_diag[f'{nfilt}p']['ugos']**2 + pass9_diag[f'{nfilt}p']['vgos']**2)
    pass9_diag[f'{nfilt}p']['Ucg'] = np.sqrt(pass9_diag[f'{nfilt}p']['ucgos']**2 + pass9_diag[f'{nfilt}p']['vcgos']**2)
    pass22_diag[f'{nfilt}p']['Ug'] = np.sqrt(pass22_diag[f'{nfilt}p']['ugos']**2 + pass22_diag[f'{nfilt}p']['vgos']**2)
    pass22_diag[f'{nfilt}p']['Ucg'] = np.sqrt(pass22_diag[f'{nfilt}p']['ucgos']**2 + pass22_diag[f'{nfilt}p']['vcgos']**2)

# save data
pickle_save('pass9_diag_calval.nc', '/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/derived_data/', pass9_diag)
pickle_save('pass22_diag_calval.nc', '/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/derived_data/', pass22_diag)


def swot_add_diagnostics_filtered(ds):
    params = dict(derivative='dxdy', n=3, min_valid_points=0.75,
                  cyclostrophy='GW', avoid_negative=False,
                  second_derivative='dxdy', kernel='circular')

    diag = []
    for i in range(len(ds.time_cycle)):
        diag.append(compute_ocean_diagnostics_from_eta(ds.adt_filtered.isel(time_cycle=i),
                                                       ds.longitude, ds.latitude, **params))

    diag_dic = {}
    for var in diag[0].keys():
        dummy = []
        for i in range(len(ds.time_cycle)):
            dummy.append(diag[i][var])
        diag_dic[var] = dummy

    dims = ds.ugos_filtered.dims
    ds = ds.assign(ugos_derived=(dims, diag_dic['ug']), vgos_derived=(dims, diag_dic['vg']),
                   ucgos_filtered=(dims, diag_dic['ucg']), vcgos_filtered=(dims, diag_dic['vcg']),
                   zeta_filtered=(dims, diag_dic['zeta']), sr_filtered=(dims, diag_dic['S']),
                   OW_filtered=(dims, diag_dic['OW']))

    ds['Ug_filtered'] = np.sqrt(ds['ugos_derived']**2 + ds['vgos_derived']**2)
    ds['Ucg_filtered'] = np.sqrt(ds['ucgos_filtered']**2 + ds['vcgos_filtered']**2)
    return ds


pass9 = swot_add_diagnostics_filtered(pass9)
pass22 = swot_add_diagnostics_filtered(pass22)

# save merged derived data
pass9.to_netcdf('/srv/data/SWOT/L3/derived_data/pass9_CalVal_v2.nc')
pass22.to_netcdf('/srv/data/SWOT/L3/derived_data/pass22_CalVal_v2.nc')
