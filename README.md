# SWOT

Repository to share preprocessing and analysis code for use of SWOT data.

## Content

### Python Scripts

**[startup_swot.py](startup_swot.py)** — Initialization script that loads common libraries (numpy, xarray, matplotlib, cartopy, gsw) and sets up the working environment for SWOT projects.

**[science_postprocess_region.py](science_postprocess_region.py)** — Main processing pipeline for SWOT L3 Science-phase Expert data that:
- Reads region-specific pass IDs
- Finds and subsets matching SWOT NetCDF files by latitude
- Computes derived fields (filtered/unfiltered ADT, geostrophic speed, vorticity, cyclogeostrophic velocity)
- Concatenates passes per cycle into merged files

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

### Step 3: Download SWOT Science Data

Use the interactive notebook to download the selected SWOT science files once you have the pass IDs and region defined in Step 2.

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

This script:
- reads the region pass ID list created in Step 2
- finds matching downloaded SWOT science NetCDF files
- subsets each file by latitude and region (to reduce data size)
- computes derived fields: filtered/unfiltered ADT, geostrophic speed, vorticity, and cyclogeostrophic velocity components
- trims passes to a common line count for concatenation
- concatenates selected passes for each cycle
- writes one merged NetCDF file per cycle for downstream analysis

A separate Cal/Val postprocessing script will be added later.


