# SWOT

Repository to share preprocessing and analysis code for use of SWOT data.

## Content

### Python Scripts

**[science_postprocess_region.py](science_postprocess_region.py)** — Main processing pipeline for SWOT L3 Science-phase Expert data that:
- reads region-specific pass IDs
- finds and subsets matching SWOT NetCDF files by latitude
- computes derived fields (filtered/unfiltered ADT, geostrophic speed, vorticity, cyclogeostrophic velocity)
- concatenates passes per cycle into merged files and saves one file per cycle

**[CalVal_postprocess_region.py](CalVal_postprocess_region.py)** — CalVal-phase processing pipeline for SWOT pass files that:
- loads and processes individual CalVal pass files
- derives filtered/unfiltered ADT and speed
- optionally adds SWOT diagnostics
- merges processed pass datasets into a single output file

**[find_swot_passes_science_Jinbo.py](find_swot_passes_science_Jinbo.py)** — Original script for finding SWOT science-phase pass IDs using regional selection criteria

### Jupyter Notebooks
(<small>adapted from from https://github.com/SWOT-community/SWOT-Oceanography</small>)

**[aviso_download_swot_interactive.ipynb](aviso_download_swot_interactive.ipynb)** — Interactive notebook for downloading SWOT data from AVISO FTP.

**[find_swot_passes_calval.ipynb](find_swot_passes_calval.ipynb)** — Identifies SWOT passes for cal/val phase.

**[find_swot_passes_science.ipynb](find_swot_passes_science.ipynb)** — Identifies SWOT passes for science phase.

### Shell Scripts
(<small>adapted from from https://github.com/SWOT-community/SWOT-Oceanography</small>)

**[download_swot_orbit.sh](download_swot_orbit.sh)** — Bash script to download SWOT orbit/SPH data files from AVISO.





## Downloading files

### Step 1: Download Orbit Files

Execute `download_swot_orbit.sh` to download necessary SWOT orbit and SPH (Sensor Performance History) files from AVISO if not already present (make sure permissions are set to allow execution on your bash file):

```bash
 ./download_swot_orbit.sh
```

This script will:
- Create a `data/` directory if it doesn't exist
- Download SWOT science nadir and swath files from AVISO
- Store files for use in subsequent processing steps

### Step 2: Find SWOT Passes for Your Region

To find the right passes numbers and fly-by times in your region use one of the two options below to identify passes for your region. **Note**: these scripts have to run with geopandas environment. 

Option A: executable script (original from SWOT community gitub)

```bash
python find_swot_passes_science_Jinbo.py
```

Option B: interactive notebook

Open and run:

- `find_swot_passes_science.ipynb` for Science-phase passes
- `find_swot_passes_calval.ipynb` for Cal/Val passes

These notebooks/scripts help you:
- search for SWOT passes in a target region
- build pass ID lists for later processing (saved as txt file)

### Step 3: Download SWOT Data

Use the interactive notebook to download the selected SWOT science files once you have the pass IDs and region defined in Step 2. For CalVal phase use other tools and define passes and cycles explicitly. 

- Open `aviso_download_swot_interactive.ipynb`
- follow the notebook cells to configure AVISO FTP credentials, target region, and file types
- ensure it reads the pass ID list for your region created in Step 2
- execute the download steps to retrieve the required data files

This notebook guides you through:
- authenticating to AVISO FTP
- reading the region pass ID list generated in Step 2
- selecting science-phase SWOT files for your region
- downloading the necessary data files for downstream processing



## Postprocessing files

### Step 4: Postprocess Science Data

Use `science_postprocess_region.py` to process downloaded SWOT Science-phase files for your selected region.  
**Note**: the script sets up a large local dask cluster for faster processing. Check resources and comment code in the script if you don't want to use it. 

This script:
- reads the region pass ID list created in Step 2
- finds matching downloaded SWOT science NetCDF files
- subsets each file by latitude and region (to reduce data size)
- computes derived fields: filtered/unfiltered ADT, geostrophic speed, vorticity, and cyclogeostrophic velocity components using SWOTdiag package by Tranchant et al.
- trims passes to a common line count for concatenation
- concatenates selected passes for each cycle
- writes one merged NetCDF file per cycle for downstream analysis

**To Run:** Execute script in command line for it to run in the background. Adding the diagnostics takes a while. You can comment the respective code if you don't need it, and the script should run well without the Dask cluster.

### Step 5: Postprocess CalVal Data

Use `CalVal_postprocess_region.py` to process SWOT CalVal files for the calibration/validation phase.

This script:
- loads CalVal pass files from the specified `--filepath`
- processes each file separately for better Dask parallelism
- derives `adt_filtered`, `adt_unfiltered`, and `speed_filtered`
- optionally adds SWOT diagnostics with `--add-diag`
- standardizes `time_cycle` coordinates and trims by latitude bounds
- concatenates processed files into a merged output dataset

**Command line example**:
```bash
python python/utils/SWOT/CalVal_postprocess_region.py \
  --filepath /srv/data/SWOT/L3/CalVal/v3_0/ \
  --passnumber 9 \
  --lat-min 30 --lat-max 55 \
  --cycle-start 474 --cycle-end 577 \
  --add-diag \
  --n-workers 10 --threads-per-worker 6 --memory-limit 20GB
```

To skip diagnostics and only perform file-level ADT/speed processing, use `--no-add-diag` argument. 


**Interactive usage**:
- open the script in a Python session or notebook
- make sure the script uses `args = parser.parse_args([])` for notebook testing
- then call `main()` directly from the notebook
- optionally edit the top-of-file defaults before running

Use `--dashboard-address :8787` if you want the Dask dashboard enabled. Then you can see the dashboard under the configured host URL, e.g. `http://localhost:8787/status` if you run locally.





