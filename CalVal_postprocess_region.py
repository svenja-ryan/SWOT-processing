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
import argparse
import sys
from pathlib import Path
sys.path.append("/home/sryan/python/")  # go to parent dir
from utils.plot_utils import finished_plot, latlon_label
from utils.datafun import datestr2doy, datetime2numeric, numeric2datetime
from utils.vector_fun import rotate, compute_angle
from utils.general import pickle_save, pickle_load

import xarray as xr
import numpy as np
from dask.distributed import Client, as_completed

# load SWOT diagnoctis
# add the directory *above* the SwotDiag/ folder
sys.path.append("/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/SwotDiag")
from SwotDiag.diagnosis import *

# load project startup file with all relevant functions
exec(open('/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/startup_swot.py').read())


# =========================================================
# Define variables
# =========================================================

# either define variables here or in command line. 
FILEPATH = '/srv/data/SWOT/L3/CalVal/v3_0/'
PASS= 9
LAT_BOUNDS = (30, 55)
CYCLE_START = 474
CYCLE_END = 577
N_WORKERS = 10
THREADS_PER_WORKER = 6
MEMORY_LIMIT = '20GB'
DASHBOARD_ADDRESS = ':8787'


#=========================================================
#%% helper functions
#=========================================================

# extract cycle numbers from filenames
def extract_cycle(files):
    import re
    cycle = []
    for file in files:
        m = re.search(r"Expert_(\d{3})", str(file))
        if not m:
            raise ValueError("Filename does not match expected pattern")
        cycle.append(int(m.groups()[0]))
    return cycle




def extract_cycle_single(file):
    import re
    m = re.search(r"Expert_(\d{3})", str(file))
    if not m:
        raise ValueError("Filename does not match expected pattern")
    return int(m.groups()[0])




# add diagnostics using SWOTDiag package
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

    # -----------------------------------------------------
    # Add metadata for vorticity
    # -----------------------------------------------------
    ds['zeta_filtered'].attrs.update({
        "units": "s-1",
        "source": "Computed using SWOTdiag package (https://github.com/treden/SwotDiag)",
        "processing_method": "9-point 2D spline fitting applied to adt_filtered"
    })
    ds['zeta_unfiltered'].attrs.update({
        "units": "s-1",
        "source": "Computed using SWOTdiag package (https://github.com/treden/SwotDiag)",
        "processing_method": "9-point 2D spline fitting applied to adt_unfiltered"
    })

    del dummy_unfilt, dummy_filt
    return ds



#=========================================================
# Merge and process pass data
#=========================================================


# def process_pass(pass_id, filepath, cycle_range=None, lat_bounds=(30, 55)):
#     """Load and preprocess a single pass by `pass_id`.

#     Returns an xarray Dataset with derived ADT, speed, diagnostics, and
#     standardized time/cycle coordinates for that file.
#     """
#     directory = Path(filepath)
#     matches = sorted(directory.glob(f"*Expert_???_{pass_id:03d}_*.nc"))
#     if len(matches) == 0:
#         raise FileNotFoundError(f"No files found for pass {pass_id} in {filepath}")

#     datasets = []
#     for f in matches:
#         dummy = xr.open_dataset(f)

#         # drop pixel/line counts if they exist to save memory 
#         if 'i_num_pixel' in dummy:
#             dummy = dummy.drop_vars(['i_num_pixel', 'i_num_line'])

#         # append to list for concatenation
#         datasets.append(dummy)

#     ds = xr.concat(datasets, dim='cycle')

#     print(f"Loaded {len(matches)} files for pass {pass_id}, total size: {ds.nbytes / 1e9:.2f} GB")


#     # derive ADT and speed
#     ds['adt_filtered'] = ds['mdt'] + ds['ssha_filtered']
#     ds['adt_unfiltered'] = ds['mdt'] + ds['ssha_unfiltered']
#     ds['speed_filtered'] = np.sqrt(ds['ugos_filtered']**2 + ds['vgos_filtered']**2)


#     # cycle numbers from filenames
#     cycles = extract_cycle(matches)
#     ds['cycle'] = (('cycle'), cycles)


#     # numeric time for interpolation, reindex cycles to requested range
#     if cycle_range is None:
#         cycle_range = np.arange(474, 578)
#     ds['timenum'] = (('cycle', 'num_lines'), datetime2numeric(ds.time, unit='s'))
#     ds = ds.reindex(cycle=cycle_range)
#     ds['time'] = (('cycle', 'num_lines'), numeric2datetime(ds.timenum, unit='s'))
#     ds = ds.assign_coords(time_cycle=("cycle", ds.time.mean(dim='num_lines').values))
#     ds = ds.swap_dims({'cycle': 'time_cycle'})

#     # limit latitude bounds to reduce data size
#     latmin, latmax = lat_bounds
#     ds = ds.where((ds.latitude > latmin) & (ds.latitude < latmax), drop=True)

#     print(f"Loaded pass {pass_id} with {len(ds.time_cycle)} cycles, lat range: {latmin} to {latmax}")



#     # compute diagnostics per cycle within this pass task
#     processed_cycles = []
#     for i in range(ds.sizes['time_cycle']):
#         ds_cycle = ds.isel(time_cycle=i)
#         ds_cycle = swot_add_diag(ds_cycle, n=9)
#         processed_cycles.append(ds_cycle)

#     ds = xr.concat(processed_cycles, dim='time_cycle')


#     print(f"Completed diagnostics for pass {pass_id}")
#     return ds




# def run_diagnostics(pass_ds, n_workers=4, threads_per_worker=4, memory_limit='20GB', dashboard_address=':8787', n=9):
#     """Run SWOT diagnostics in parallel across time_cycle slices."""

#     print("Running diagnostics with dask now")

#     with Client(
#             n_workers=n_workers,
#             threads_per_worker=threads_per_worker,
#             memory_limit=memory_limit,
#             dashboard_address=dashboard_address
#         ) as client:
#         print(client)

#         futures = [
#             client.submit(swot_add_diag, pass_ds.isel(time_cycle=i), n=n)
#             for i in range(pass_ds.sizes['time_cycle'])
#         ]

#         results = []
#         bad = []

        # for future in as_completed(futures):
        #     try:
        #         results.append(future.result())
        #     except Exception as e:
        #         bad.append((future.key, repr(e)))
        #         print(f"Failed task: {future.key}")
        #         print(f"  Error: {repr(e)}")

        # if bad:
        #     print(f"{len(bad)} tasks failed")
        #     raise RuntimeError("Some Dask tasks failed. See messages above.")

        # processed = xr.concat(results, dim='time_cycle')
        # print(f"Processed {len(results)} time steps in parallel")
        # return processed




def process_pass(f, lat_bounds=(30, 55),add_diag=False):
    """Load and preprocess a single pass by `pass_id`.

    Returns an xarray Dataset with derived ADT, speed, diagnostics, and
    standardized time/cycle coordinates for that file.
    """

    # open dataset
    try:
        ds = xr.open_dataset(f).drop(
                ['dac', 'calibration', 'ocean_tide', 'cross_track_distance', 'internal_tide', 'sigma0'],
                errors='ignore'
            )
        
        print(f"loaded file {f}")
        
        # drop pixel/line counts if they exist to save memory 
        if 'i_num_pixel' in ds:
            ds = ds.drop_vars(['i_num_pixel', 'i_num_line'])

        # cycle numbers from filenames
        cycle = extract_cycle_single(f)
        ds['cycle'] = (('cycle'), [cycle])

        # derive ADT and speed
        ds['adt_filtered'] = ds['mdt'] + ds['ssha_filtered']
        ds['adt_unfiltered'] = ds['mdt'] + ds['ssha_unfiltered']
        ds['speed_filtered'] = np.sqrt(ds['ugos_filtered']**2 + ds['vgos_filtered']**2)

        # limit latitude bounds to reduce data size
        latmin, latmax = lat_bounds
        ds = ds.where((ds.latitude > latmin) & (ds.latitude < latmax), drop=True)

        # compute diagnostics 
        if add_diag:
            ds = swot_add_diag(ds, n=9)

        return {"ok": True, "data": ds.load(), "file": str(f)}

    except Exception as e:
        return {"ok": False, "error": str(e), "file": str(f)}
   
 

#==========================================================
#%% main function to run the full workflow for specified passes and parameters
#==========================================================

# can be run from command line with arguments, e.g.:
# python CalVal_postprocess_region.py --filepath /srv/data/SWOT/L3/CalVal/v3_0/ --passes 9 22 --lat-min 30 --lat-max 55 --n-workers 10 --threads-per-worker 6

def main():
    parser = argparse.ArgumentParser(description='CalVal SWOT pass postprocessing')
    parser.add_argument('--filepath', default=FILEPATH, help='Directory containing SWOT files')
    parser.add_argument('--passes', type=int, default=PASS, help='Pass IDs to process')
    parser.add_argument('--lat-min', type=float, default=LAT_BOUNDS[0], help='Minimum latitude bound')
    parser.add_argument('--lat-max', type=float, default=LAT_BOUNDS[1], help='Maximum latitude bound')
    parser.add_argument('--cycle-start', type=int, default=CYCLE_START, help='Starting cycle for reindexing')
    parser.add_argument('--cycle-end', type=int, default=CYCLE_END, help='Ending cycle for reindexing')
    parser.add_argument('--n-workers', type=int, default=N_WORKERS, help='Number of Dask workers')
    parser.add_argument('--threads-per-worker', type=int, default=THREADS_PER_WORKER, help='Threads per Dask worker')
    parser.add_argument('--memory-limit', default=MEMORY_LIMIT, help='Memory limit per Dask worker')
    parser.add_argument('--dashboard-address', default=DASHBOARD_ADDRESS, help='Dask dashboard address')
    args = parser.parse_args()

    lat_bounds = (args.lat_min, args.lat_max)
    cycle_range = np.arange(args.cycle_start, args.cycle_end + 1)


    # -----------------------------------------------------
    # Start local Dask cluster
    # -----------------------------------------------------
    with Client(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
        dashboard_address=args.dashboard_address
    ) as client:
        print(client)

        directory = Path(args.filepath)
        matches = sorted(directory.glob(f"*Expert_???_{args.passes:03d}_*.nc"))
        if len(matches) == 0:
            raise FileNotFoundError(f"No files found for pass {args.passes} in {args.filepath}")
        
        # -------------------------------------------------
        # Submit all processing tasks
        # -------------------------------------------------
        futures = [client.submit(process_pass,str(f),lat_bounds,add_diag=True) for f in matches]
        # -------------------------------------------------

        ds_list = []
        bad_files = []

        # -------------------------------------------------
        # Collect results as tasks finish
        # -------------------------------------------------
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["ok"]:
                    ds_list.append(result["data"])
            else:
                bad_files.append(result)
                print(f"Skipped bad file: {result['file']}")
                print(f"  Error: {result['error']}")

        # -----------------------------------------------------
        # Concatenate processed datasets
        # -----------------------------------------------------
        ds = xr.concat(ds_list, dim="cycle")

        # -----------------------------------------------------
        # Concatenate processed datasets
        # -----------------------------------------------------
        # numeric time for interpolation, reindex cycles to requested range
        if cycle_range is None:
            cycle_range = np.arange(474, 578)
        ds['timenum'] = (('cycle', 'num_lines'), datetime2numeric(ds.time, unit='s'))
        ds = ds.reindex(cycle=cycle_range)
        ds['time'] = (('cycle', 'num_lines'), numeric2datetime(ds.timenum, unit='s'))
        ds = ds.assign_coords(time_cycle=("cycle", ds.time.mean(dim='num_lines').values))
        ds = ds.swap_dims({'cycle': 'time_cycle'})

        outfile = f'/srv/data/SWOT/L3/derived_data/pass{args.passes}_CalVal_processed.nc'
        ds.to_netcdf(outfile)
        print(f"Saved processed pass {args.passes} to {outfile}")

    print("Processing complete for all passes.")


if __name__ == '__main__':
    main()






