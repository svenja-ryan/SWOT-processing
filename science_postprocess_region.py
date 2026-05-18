#!/usr/bin/env python
# coding: utf-8
#%%  Postprocess SWOT Science phase data for one region
# =========================================================
#
# This script postprocesses SWOT L3 Science-phase Expert data
# for a selected geographic region and cycle range.
#
# For each SWOT cycle, the script:
#   1. Reads a region-specific list of pass IDs.
#   2. Finds matching SWOT Expert NetCDF files.
#   3. Subsets each file by latitude to reduce size
#   4. Computes derived fields:
#        - filtered ADT
#        - unfiltered ADT
#        - filtered geostrophic speed
#        - relative vorticity from filtered ADT
#        - relative vorticity from unfiltered ADT
#        - cyclogeostrophic velocity components
#  #   6. Trims files to a common num_lines length to allow concatenation.
#   7. Concatenates all selected passes for each cycle.
#   8. Adds useful coordinates:
#        - time_pass
#        - cycle
#        - passID
#   9. Writes one merged NetCDF file per cycle.
#
#
# =========================================================
# Required inputs
# =========================================================
# 1. SWOT L3 Science Expert NetCDF files:
#      /srv/data/SWOT/L3/Science/v3_0/
#
# 2. Region-specific passID file:
#      ./data_download/passID_files/passIDs_<region>_science*.txt
#
#    The passID file should contain one pass ID per row, with a
#    one-line header skipped by np.loadtxt(..., skiprows=1).
#
# 3. Local SwotDiag package:
#      /home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/SwotDiag/
#
#
# =========================================================
# User-defined settings
# =========================================================
# region:
#     Region name used to find the passID file and define output folder.
#
# latmin, latmax:
#     Latitude limits used to subset each SWOT swath.
#
# nlines_max:
#     Maximum number of num_lines retained from each file.
#     This is currently hard-coded so all files can be concatenated.
#
# start_cycle, end_cycle:
#     First and last SWOT cycles to process.
#
# infile_pattern:
#     Glob pattern for finding the region-specific passID file.
#
# swot_dir:
#     Directory containing input SWOT L3 Science Expert files.
#
# outdir:
#     Directory where merged cycle files will be written.
#
# =========================================================
# Command-line use
# =========================================================
# The script can be run from the command line with optional arguments for cycle range:
#   python postprocess_swot_region.py --start-cycle 31 --end-cycle 49
#
# If no arguments are provided, it defaults to processing cycles define in script.
#
#
#
# =========================================================
# To-Do
# =========================================================
# - [ ] add section to detect shortes num_lines in file so that it does not have to be defined (makes it more applicable to other regions)
# - [ ] create version to parce input in command line
# - [ ] define





# =========================================================
#%% Start work
# =========================================================
import sys
import os
from pathlib import Path
import glob
import re
import numpy as np
import xarray as xr
from dask.distributed import Client, as_completed

# Change the notebook’s working directory to a known location
os.chdir('/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/')

# ---------------------------------------------------------
# Imports from local packages
# ---------------------------------------------------------
sys.path.insert(0, "/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024")
from SwotDiag.SwotDiag.diagnosis import *


# =========================================================
# Define variables
# =========================================================
region = "NWA"  # Has to have passID file
latmin, latmax = 30,47    # Ross -78,-68
nlines_max = 1029 # this is based in min len of lines depending on region (not sure how to fix it yet)
infile_pattern = f"./data_download/passID_files/passIDs_{region}_science*.txt"
swot_dir = Path("/srv/data/SWOT/L3/Science/v3_0/")
outdir = f"/srv/data/SWOT/L3/derived_data/{region}/"

#Default cycle range, used only if not provided at command line
start_cycle = 49
end_cycle = 50
# =========================================================


# =========================================================
# Set up option for command line input
# =========================================================
def parse_args():
    """Parse command-line inputs."""
    parser = argparse.ArgumentParser(
        description="Postprocess SWOT Science-phase data for one region."
    )

    parser.add_argument(
        "--start-cycle",
        type=int,
        default=start_cycle,
        help="First SWOT cycle to process, e.g. 31"
    )

    parser.add_argument(
        "--end-cycle",
        type=int,
        default=end_cycle,
        help="Last SWOT cycle to process, e.g. 49"
    )

    return parser.parse_args()
# =========================================================





# =========================================================
#%% Helper functions
# =========================================================
def extract_passID(fname):
    """Extract passID from filename pattern '*Expert_???_<passID>_*.nc'."""
    m = re.search(r"Expert_\d{3}_(\d{3})_", fname.name)
    if not m:
        return None
    return int(m.group(1))


def extract_cycle(files):
    """Extract cycle and passID from filenames."""
    cycle = []
    passID = []
    for file in files:
        m = re.search(r"Expert_(\d{3})_(\d{3})", str(file))
        cycle.append(int(m.groups()[0]))
        passID.append(int(m.groups()[1]))
    return cycle, passID


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


def process_swot_file(f, latmin, latmax, n_zeta=9, nlines_max=1029):
    """Open, subset, derive diagnostics, and return one processed SWOT file."""
    try:
        dummy = xr.open_dataset(f).drop(
            ['dac', 'calibration', 'ocean_tide', 'cross_track_distance', 'internal_tide', 'sigma0']
        )

        # limit to lat range to reduce file size and speed up processing
        dummy = dummy.where((dummy.latitude > latmin) & (dummy.latitude < latmax), drop=True)

        # derive ADT and speed
        dummy["adt_filtered"] = dummy["mdt"] + dummy["ssha_filtered"]
        dummy["adt_unfiltered"] = dummy["mdt"] + dummy["ssha_unfiltered"]
        dummy["speed_filtered"] = np.sqrt(dummy["ugos_filtered"]**2 + dummy["vgos_filtered"]**2)

        # compute vorticity and cyclogeostrophic velocities using SwotDiag package  
        dummy = swot_add_diag(dummy, n=n_zeta)

        #
        drop_vars = [v for v in ["i_num_pixel", "i_num_line"] if v in dummy.variables]
        if drop_vars:
            dummy = dummy.drop_vars(drop_vars)
        # trim to common num_lines length so all files can be concatenated
        if dummy.sizes["num_lines"] > nlines_max:
            dummy = dummy.isel(num_lines=slice(0, nlines_max))

        return {"ok": True, "data": dummy.load(), "file": str(f)}

    except Exception as e:
        return {"ok": False, "error": str(e), "file": str(f)}




# =========================================================
#%% Main workflow
# =========================================================
def main():

    # command line input for cycle range
    args = parse_args()
    start_cycle_run = args.start_cycle
    end_cycle_run = args.end_cycle

    print(f"Processing cycles {start_cycle_run} to {end_cycle_run} for region {region} with lat range {latmin} to {latmax}")

    # -----------------------------------------------------
    # Start local Dask cluster
    # -----------------------------------------------------
    with Client(
        n_workers=10,
        threads_per_worker=6,
        memory_limit='20GB',
        dashboard_address=':8787'
    ) as client:

        print(client)

        # -----------------------------------------------------
        # Load passIDs
        # -----------------------------------------------------
        files = glob.glob(infile_pattern)
        if not files:
            raise FileNotFoundError(f"No files match {infile_pattern}")

        passIDs = np.loadtxt(files[0], skiprows=1, dtype=int)

        for cyc in range(start_cycle_run, end_cycle_run+1):
            cyc_str = f"{cyc:03d}"
            files_cycle = sorted(swot_dir.glob(f"*Expert_{cyc_str}_*.nc"))

            filtered_files = [f for f in files_cycle if extract_passID(f) in passIDs]
            print(f"Found {len(filtered_files)} files matching passIDs out of {len(files_cycle)} for cycle {cyc_str}")


            # -------------------------------------------------
            # Submit all processing tasks
            # -------------------------------------------------
            futures = [client.submit(process_swot_file, f, latmin, latmax) for f in filtered_files]

            ds_list = []
            bad_files = []
            n_total = len(futures)

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

                if i % 50 == 0 or i == n_total:
                    print(f"Processed {i}/{n_total} files ({100*i/n_total:.1f}%)")

            # -----------------------------------------------------
            # Concatenate processed datasets
            # -----------------------------------------------------
            # first make sure they are consistent in size (num_lines sometimes varies slightly)
            # num_lines = min([ds.sizes["num_lines"] for ds in ds_list])
            # ds_list = []
            # ds_list = 
            ds_merged = xr.concat(ds_list, dim="passes")

            # -----------------------------------------------------
            # Add time coordinate for easier selection
            # -----------------------------------------------------
            time_pass = ds_merged.time.mean(('num_lines', 'num_pixels'))
            ds_merged = ds_merged.assign_coords(time_pass=("passes", time_pass.values))

            # -----------------------------------------------------
            # Add cycle and passID coordinates
            # -----------------------------------------------------
            cycle, passID = extract_cycle(filtered_files)
            ds_merged = ds_merged.assign_coords(cycle=("passes", cycle))
            ds_merged = ds_merged.assign_coords(passID=("passes", passID))

            print("data merged")

            # -----------------------------------------------------
            # Add metadata for vorticity
            # -----------------------------------------------------
            ds_merged['zeta_filtered'].attrs.update({
                "units": "s-1",
                "source": "Computed using SWOTdiag package (https://github.com/treden/SwotDiag)",
                "processing_method": "9-point 2D spline fitting applied to adt_filtered"
            })
            ds_merged['zeta_unfiltered'].attrs.update({
                "units": "s-1",
                "source": "Computed using SWOTdiag package (https://github.com/treden/SwotDiag)",
                "processing_method": "9-point 2D spline fitting applied to adt_unfiltered"
            })

            # -----------------------------------------------------
            # Save merged dataset
            # -----------------------------------------------------
            print("saving data now")
            ds_merged.to_netcdf(outdir+f"{region}_merged_{latmin}_{latmax}_cycle{cyc_str}_"\
                                f"{pd.to_datetime(time_pass).strftime('%Y-%m-%d')[0]}"\
                                    f"_{pd.to_datetime(time_pass).strftime('%Y-%m-%d')[-1]}.nc")   # safe cycle start & end in filename
            print("file saved")



# only runs main function when file is executed directly
if __name__ == "__main__":
    main()

