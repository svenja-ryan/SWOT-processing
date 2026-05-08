# ----------------------------------------------------------
# Load modules
# ----------------------------------------------------------
import os
import sys
import string
import warnings
import datetime as dt

# Numerical and data handling
import numpy as np
import pandas as pd
import xarray as xr
import scipy.io as sc
import scipy.signal as signal
from scipy.interpolate import griddata

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

#---------------------
#
#-----------------------
# Cartopy for geospatial plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Other third‐party libraries
import cmocean as cm  # Oceanographic colormaps
import gsw           # TEOS-10 Gibbs Seawater functions

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

sys.path.append("/home/sryan/python/") # go to parent dir
from utils.general import equal_distance_line


# ---------------------------------------------------------------------
# Working Directory
# ---------------------------------------------------------------------
# Change the notebook’s working directory to a known location
os.chdir('/home/sryan/python/projects/NASA_SWOT_NWA_shelf_2024/')


# ---------------------------------------------------------------------
# Conversion Constants
# ---------------------------------------------------------------------
# Centimeters per inch (for converting figure sizes)
CM_PER_INCH = 1 / 2.54


# ---------------------------------------------------------------------
# Load Bathymetry Data
# ---------------------------------------------------------------------
# Subset East Coast: longitude –85 to –40, latitude 25 to 55
bathy = (
    xr.open_dataset('/vast/clidex/data/bathymetry/ETOPO1/ETOPO1_Bed_g_gmt4.grd')
      .sel(x=slice(-85, -30), y=slice(25, 55))
      .load()
)

# ---------------------------------------------------------------------
# extract ice shelves for plotting
# ---------------------------------------------------------------------
from cartopy.feature import NaturalEarthFeature
ice_shelves = NaturalEarthFeature(
    category='physical',
    name='antarctic_ice_shelves_polys',
    scale='10m',
    facecolor='lightblue',
    edgecolor='black',
    alpha=0.5
)

#-----------------------------------------------------------------------
# map extents for regions
# ----------------------------------------------------------------------
map_extent={}
map_extent = {
    "Weddell": {"lon": [-62, 0], "lat": [-80, -65]},
    "Ross": {"lon": [160, -150], "lat": [-78, -70]},
    "Amery": {"lon": [60, 80], "lat": [-73, -63]},
}

# ---------------------------------------
# define map extends for each canyon
# ---------------------------------------
extent_Hudson = [-72.8, -71.8, 39, 40]
extent_Baltimore = [-74, -73, 37.8, 38.7]
extent_MAB = [-73, -71.5, 38.5, 40]
extent_all = [-75.5, -70.5, 36, 41]  # slightly more canyone focused than full MAB

# ---------------------------------------------------------------------
# Fixed Instrument/Location Coordinates
# ---------------------------------------------------------------------

# Magdalena PIES (P1–P5), plus C1 and C2 reference stations
pies_calval = {
    'P1': (36.2334, -74.3662),
    'P2': (35.9997, -74.5339),
    'P3': (36.0039, -74.1969),
    'P4': (36.0001, -74.6728),
    'P5': (36.0001, -74.3671),
    'C1': (36.0521, -74.7061),
    'C2': (35.7001, -74.77015),
}

# Pioneer II arrow moorings: (latitude, longitude, depth_meters)
pioneer = {
    'CP12WESW': (35.95,   -75.3333, 30.0),
    'CP10CNSM': (35.95,   -75.1250, 30.0),
    'CP12CNSW': (35.95,   -75.1250, 30.0),
    'CP11NOSM': (36.175,  -74.8267, 100.0),
    'CP13NOPM': (36.175,  -74.8267, 100.0),
    'CP14NEPM': (36.0536, -74.7776, 300.0),
    'CP13EAPM': (35.95,   -74.8457, 100.0),
    'CP14SEPM': (35.8514, -74.8482, 300.0),
    'CP11SOSM': (35.725,  -74.8530, 100.0),
    'CP13SOPM': (35.725,  -74.8530, 100.0),
}

# Oleander line endpoints: (latitude, longitude)
oleander = {
    'NJ': (40.6597, -74.2014),
    'BM': (32.2951, -64.7842),
}


# ---------------------------------------------------------------------
# Plotting Configuration
# ---------------------------------------------------------------------
SMALL_SIZE  = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font',    size=SMALL_SIZE, serif='Helvetica Neue', weight='normal')
plt.rc('text',    usetex='false')
plt.rc('axes',    titlesize=MEDIUM_SIZE, labelsize=SMALL_SIZE, labelweight='normal', facecolor='white')
plt.rc('xtick',   labelsize=SMALL_SIZE)
plt.rc('ytick',   labelsize=SMALL_SIZE)
plt.rc('legend',  fontsize=SMALL_SIZE, frameon=False)
plt.rc('figure',  titlesize=MEDIUM_SIZE, titleweight='bold', autolayout=True)
plt.rc('path',    simplify=True)
plt.rc('axes.spines', top=False, right=False)
plt.rcParams['figure.figsize'] = (8, 4)  # Default figure size (in inches)


def font_for_print():
    """
    Set font sizes and weights for print‐style figures.
    """
    SMALL = 6
    MEDIUM = 8
    plt.rc('font',    size=SMALL, serif='Helvetica Neue', weight='normal')
    plt.rc('text',    usetex='false')
    plt.rc('axes',    titlesize=MEDIUM, labelsize=SMALL, labelweight='normal')
    plt.rc('xtick',   labelsize=SMALL)
    plt.rc('ytick',   labelsize=SMALL)
    plt.rc('legend',  fontsize=SMALL, frameon=False)
    plt.rc('figure',  titlesize=MEDIUM, titleweight='bold', autolayout=True)

def font_MEDIUM():
    """
    Set font sizes and weights for print‐style figures.
    """
    SMALL = 8
    MEDIUM = 10
    plt.rc('font',    size=SMALL, serif='Helvetica Neue', weight='normal')
    plt.rc('text',    usetex='false')
    plt.rc('axes',    titlesize=MEDIUM, labelsize=SMALL, labelweight='normal')
    plt.rc('xtick',   labelsize=SMALL)
    plt.rc('ytick',   labelsize=SMALL)
    plt.rc('legend',  fontsize=SMALL, frameon=False)
    plt.rc('figure',  titlesize=MEDIUM, titleweight='bold', autolayout=True)


def font_for_pres():
    """
    Set font sizes and weights for presentation‐style figures.
    """
    SMALL = 10
    MEDIUM = 12
    plt.rc('font',    size=SMALL, serif='Helvetica Neue', weight='normal')
    plt.rc('text',    usetex='false')
    plt.rc('axes',    titlesize=MEDIUM, labelsize=SMALL, labelweight='normal')
    plt.rc('xtick',   labelsize=SMALL)
    plt.rc('ytick',   labelsize=SMALL)
    plt.rc('legend',  fontsize=SMALL, frameon=False)
    plt.rc('figure',  titlesize=MEDIUM, titleweight='bold', autolayout=True)



def map_init(ax=None,extent=[-76, -70.5, 36, 41]):
    """
    Plot one or more xarray datasets (`datasets`) of a scalar variable `var`
    over the Mid‐Atlantic Bight region. Adds GSHHS coastline, 100 m bathymetry,
    gridlines, and an optional colorbar if any data were plotted.
    """
    font_for_pres()
    # Create new figure & PlateCarree axis if needed
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 5),
            subplot_kw=dict(projection=ccrs.PlateCarree())
        )
    else:
        fig = ax.figure  # <-- IMPORTANT
        
    if extent is None:
        ax.set_extent(extent,crs=ccrs.PlateCarree())
    else:
        ax.set_extent(extent,crs=ccrs.PlateCarree())

    # Track last mappable (for colorbar)
    cc = None

    # Add gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=1, color='lightgray', alpha=0.5, linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False

    # Add GSHHS coastline
    ax.add_feature(
        cfeature.GSHHSFeature(scale='high'),
        facecolor='lightgray', edgecolor='None'
    )

    # Contour 100 m bathymetry
    ax.contour(
        bathy.x, bathy.y, bathy.z * (-1),
        levels=[50,75,100,200, 1000], colors='k',
        transform=ccrs.PlateCarree(),
        linewidths=1, alpha=0.3
    )

    return fig, ax


# ---------------------------------------------------------------------
# Plotting Routines
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Table of Contents (Plotting Functions)
# ---------------------------------------------------------------------
# 1. plot_calval           → Plot scalar fields (e.g., salinity, temperature) over MAB region,
#                            with coastline, bathymetry contour, gridlines, and optional colorbar.
# 2. plot_calval_quiver    → Overlay quiver (u, v) vectors on a PlateCarree axis,
#                            with optional downsampling (inc2d) to avoid overcrowding.
# 3. plot_viirs            → Fetch and plot VIIRS SST for a given date (YYYY-MM-DD),
#                            including gridlines, coastline, bathymetry, and a horizontal colorbar.
# 4. plot_globcolour       → Fetch and plot GlobColour CHLa for a given date (YYYY-MM-DD),
#                            including gridlines, coastline, bathymetry, and a horizontal colorbar.
# 5. plot_mursst            → Fetch and plot MURSST for a given date (YYYY-MM-DD),
#                            including gridlines, coastline, bathymetry, and a horizontal colorbar.
# 6. zoom_shelfbreak_MAB   → Zoom an existing axis to the shelf-break region in the MAB.
# 7. plot_oleander         → Draw the Oleander monitoring line between New Jersey and Bermuda.
# 8. plot_pies_calval      → Mark Magdalena PIES (P1–P5) positions on a map axis.
# 9. plot_pioneer2         → Mark Pioneer II arrow mooring positions on a map axis.
# 10. font_for_print        → Configure Matplotlib font settings for print-style figures.
# 11. font_for_pres         → Configure Matplotlib font settings for presentation-style figures.
# 12. iceshelves           -> creates ice shelf mask for Antarctica to plot

def plot_calval(datasets, var=None, vminmax=None, title=None, cmap=None, ax=None,extent=None):
    """
    Plot one or more xarray datasets (`datasets`) of a scalar variable `var`
    over the Mid‐Atlantic Bight region. Adds GSHHS coastline, 100 m bathymetry,
    gridlines, and an optional colorbar if any data were plotted.
    """
    font_for_pres()
    # Create new figure & PlateCarree axis if needed
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 5),
            subplot_kw=dict(projection=ccrs.PlateCarree())
        )
    else:
        fig = ax.figure  # <-- IMPORTANT
        
    if extent is None:
        ax.set_extent([-78, -70, 35, 42],crs=ccrs.PlateCarree())
    else:
        ax.set_extent(extent,crs=ccrs.PlateCarree())

    # Track last mappable (for colorbar)
    cc = None

    # Plot each dataset
    if datasets:
        for ds in datasets:
            cc = ds[var].plot(
                ax=ax,
                x='longitude',
                y='latitude',
                vmin=vminmax[0], vmax=vminmax[1],
                cmap=cmap, add_colorbar=False
            )

    # Add gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=1, color='lightgray', alpha=0.5, linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False

    # Add GSHHS coastline
    ax.add_feature(
        cfeature.GSHHSFeature(scale='high'),
        facecolor='lightgray', edgecolor='None'
    )

    # Contour 100 m bathymetry
    ccbathy=ax.contour(
        bathy.x, bathy.y, bathy.z * (-1),
        levels=[100], colors='k',
        transform=ccrs.PlateCarree(),
        linewidths=1, alpha=0.3
    )
    ax.clabel(ccbathy,fontsize=8)
    #     # Contour 100 m bathymetry
    # ccbathy=ax.contour(
    #     bathy.x, bathy.y, bathy.z * (-1),
    #     levels=[200], colors='r',
    #     transform=ccrs.PlateCarree(),
    #     linewidths=2, alpha=0.5
    # )

    ax.set_title(title)

    # Create a horizontal colorbar only if data were plotted
    ax_cbar = None
    if cc is not None:
        ax_cbar = ax.inset_axes([0.13, 0.94, 0.3, 0.02])
        plt.colorbar(cc, cax=ax_cbar, label=var, orientation='horizontal')

    if fig is not None:
        return fig, ax
    return fig, ax


def plot_calval_quiver(datasets, inc2d=(1, 1), alpha=1, ax=None, color='k', scale=40,extent=None):
    """
    Plot quiver arrows (u, v) from each dataset in `datasets` on a PlateCarree axis.
    Only every inc2d[0]-th longitude point and inc2d[1]-th latitude point are plotted.
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 5),
            subplot_kw=dict(projection=ccrs.PlateCarree())
        )
    if extent is None:
        ax.set_extent([-78, -70, 35, 42])

    inc_lon, inc_lat = inc2d
    for ds in datasets:
        ax.quiver(
            ds.longitude[0:-1:inc_lon, 0:-1:inc_lat].values,
            ds.latitude[  0:-1:inc_lon, 0:-1:inc_lat].values,
            ds.ugos_filtered[0:-1:inc_lon, 0:-1:inc_lat].values,
            ds.vgos_filtered[0:-1:inc_lon, 0:-1:inc_lat].values,
            scale=scale, scale_units='width',
            transform=ccrs.PlateCarree(),
            alpha=alpha, color=color
        )



def plot_globcolour(timestr,vmin=0.2, vmax=3,ax=None,extent=None,cax=None):
    """
    Plot Chlorophyll-a from GlobColour dataset
    """
    fig = None
    # if no ax handle is given then initiate plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(projection=ccrs.PlateCarree()))
    if extent is None:
        extent = [-78, -70, 35, 42]
        ax.set_extent(extent)
    
    # 1) open dataset
    ds = xr.open_dataset(f'/vast/clidex/data/obs/GlobColour/daily/cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D_CHL-flags_85.06W-35.02W_22.10N-47.98N_{timestr}.nc')

    # 2) Plot CHLa without auto‐colorbar, then add grid, coast, bathy, title, colorbar
    # Choose a positive vmin (cannot be zero) and your desired vmax
    # vmin = 0.2   # lowest nonzero value on the log scale
    # vmax = 3
    # Create a LogNorm between vmin and vmax
    log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cc= ds['CHL'].plot(ax=ax, norm=log_norm, cmap='cmo.haline',
                   add_colorbar=False, transform=ccrs.PlateCarree())
    # cc= ds['CHL'].plot(ax=ax, cmap='cmo.haline',
    #                add_colorbar=False, transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)

    # 3) make pretty
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color="lightgray", alpha=0.2, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    ax.add_feature(cfeature.GSHHSFeature(scale="high"),
                   facecolor="lightgray", edgecolor="None")
    ax.contour(bathy.x, bathy.y, bathy.z * -1, levels=[100], colors="k",
               transform=ccrs.PlateCarree(), linewidths=1, alpha=0.3)
    ax.set_title(timestr)
    # cax = ax.inset_axes([0.03, 0.94, 0.3, 0.02])
    # # make sure ticks are equally spaced 
    # # compute 5 ticks equally spaced in log10-space
    # log_min, log_max = np.log10(vmin), np.log10(vmax)
    # ticks_log = np.linspace(log_min, log_max, 5)
    # ticks     = 10**ticks_log
    # cbar = plt.colorbar(cc, cax=cax, label='Chlorophyll-a [mg/m$^3$]', orientation="horizontal",ticks=ticks)
    # cbar.ax.xaxis.set_tick_params(length=0)
    if cax is None:
        cax = ax.inset_axes([0.13, 0.94, 0.3, 0.02])
    plt.colorbar(cc, cax=cax, label='Chlorophyll-a [mg/m$^3$]', orientation="horizontal")
    
    if fig:
        return fig, ax



def plot_viirs(timestr, vmin, vmax, ax=None, extent=None):
    """
    Plot VIIRS SST for a given date string `timestr` (YYYY-MM-DD).
    Opens the nearest 3-day ACSPOCW file, then plots SST on a PlateCarree axis
    with colormap 'RdYlBu_r'. Includes GSHHS coastline, 100 m bathymetry contour,
    gridlines, and a horizontal colorbar.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 5),
            subplot_kw=dict(projection=ccrs.PlateCarree())
        )
    if extent is None:
        extent = [-78, -70, 35, 42]
    ax.set_extent(extent)

    # Convert timestr to day-of-year, find nearest 3-day SST file
    doy = datestr2doy(timestr)
    doy_file = np.arange(1, 367, 1)
    closest = doy_file[np.abs(doy_file - doy).argmin()]
    ds = xr.open_dataset(
        f"/vast/clidex/data/obs/SST/NOAAVIIRS/daily/"
        #f"ACSPOCW_{timestr[0:4]}{closest:03d}_3DAY_MULTISAT_SST-NGT_EC_750M.nc4
        f"ACSPOCW_{timestr[0:4]}{closest:03d}_DAILY_MULTISAT_SST-NGT_EC_750M.nc4"
    )

    # Plot SST (no auto colorbar), add grid, coast, bathy, and title
    qm = ds["sst"].squeeze().plot(
        ax=ax, x="lon", y="lat",
        vmin=vmin, vmax=vmax,
        cmap='RdYlBu_r', add_colorbar=False
    )
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=1, color="lightgray", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.add_feature(
        cfeature.GSHHSFeature(scale="high"),
        facecolor="lightgray", edgecolor="None"
    )
    ax.contour(
        bathy.x, bathy.y, bathy.z * -1,
        levels=[100], colors="k",
        transform=ccrs.PlateCarree(),
        linewidths=1, alpha=0.3
    )
    ax.set_title(timestr)

    # Horizontal colorbar above the axis
    cax = ax.inset_axes([0.13, 0.94, 0.3, 0.02])
    plt.colorbar(qm, cax=cax, label="sea surface temperature [°C]", orientation="horizontal")

    if fig:
        return fig, ax



def plot_avhrr(ds, timestr, vmin, vmax, ax=None, extent=None):
    """
    Plot AVHRR SST data
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 5),
            subplot_kw=dict(projection=ccrs.PlateCarree())
        )
    if extent is None:
        extent = [-78, -70, 35, 42]
    ax.set_extent(extent)


    # Plot SST (no auto colorbar), add grid, coast, bathy, and title
    qm = ds["mcsst"].sel(time=timestr,method='nearest').squeeze().plot(
        ax=ax, x="lon", y="lat",
        vmin=vmin, vmax=vmax,
        cmap='RdYlBu_r', add_colorbar=False
    )
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=1, color="lightgray", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.add_feature(
        cfeature.GSHHSFeature(scale="high"),
        facecolor="lightgray", edgecolor="None"
    )
    ax.contour(
        bathy.x, bathy.y, bathy.z * -1,
        levels=[100], colors="k",
        transform=ccrs.PlateCarree(),
        linewidths=1, alpha=0.3
    )
    ax.set_title(timestr)

    # Horizontal colorbar above the axis
    cax = ax.inset_axes([0.13, 0.94, 0.3, 0.02])
    plt.colorbar(qm, cax=cax, label="sea surface temperature [°C]", orientation="horizontal")

    if fig:
        return fig, ax




def plot_mursst(timestr, vmin=None, vmax=None, ax=None, extent=None,cmap='RdYlBu_r'):
    """
    Plot MURSST for a given date string `timestr` (YYYYMMDD).
    Opens file mathing timestr, then plots SST on a PlateCarree axis
    with colormap 'RdYlBu_r'. Includes GSHHS coastline, 100 m bathymetry contour,
    gridlines, and a horizontal colorbar.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6, 5),
            subplot_kw=dict(projection=ccrs.PlateCarree())
        )
    if extent is None:
        extent = [-78, -70, 35, 42]
    ax.set_extent(extent)

    # Convert timestr to day-of-year, find nearest 3-day SST file
    ds = xr.open_dataset(f'/vast/clidex/data/obs/SST/MURSST/data/{timestr}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc').sel(lon=slice(extent[0]-0.5,extent[1]+0.5),lat=slice(extent[2]-0.5,extent[3]+0.5)).squeeze()

    # Plot SST (no auto colorbar), add grid, coast, bathy, and title
    if vmin:
        qm = (ds["analysed_sst"]-273.15).squeeze().plot(
            ax=ax, x="lon", y="lat",
            vmin=vmin, vmax=vmax,
            cmap=cmap, add_colorbar=False
        )
    if vmin is None:
        qm = (ds["analysed_sst"]-273.15).squeeze().plot.pcolormesh(
            ax=ax, x="lon", y="lat",
            cmap=plt.get_cmap(cmap,30), add_colorbar=False
        )
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=1, color="lightgray", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.add_feature(
        cfeature.GSHHSFeature(scale="high"),
        facecolor="lightgray", edgecolor="None"
    )
    ax.contour(
        bathy.x, bathy.y, bathy.z * -1,
        levels=[100,1000], colors="k",
        transform=ccrs.PlateCarree(),
        linewidths=1, alpha=0.3
    )
    ax.set_title(timestr)

    # Horizontal colorbar above the axis
    cax = ax.inset_axes([0.13, 0.94, 0.3, 0.02])
    cbar = plt.colorbar(qm, cax=cax, label="sea surface temperature [°C]", orientation="horizontal")
    # tell the colorbar to use at most 5 ticks
    cbar.locator = mticker.MaxNLocator(nbins=5)
    cbar.update_ticks()

    if fig:
        return fig, ax


def zoom_shelfbreak_MAB(ax):
    """
    Zoom the given axis to the MAB shelf‐break region.
    """
    ax.set_extent([-76, -71, 36, 40])


def plot_oleander(ax, linestyle='--', color='k'):
    """
    Plot the Oleander line between New Jersey (NJ) and Bermuda (BM).
    """
    lat1, lon1 = oleander['NJ']
    lat2, lon2 = oleander['BM']
    ax.plot([lon1, lon2], [lat1, lat2], linestyle=linestyle, color=color)


def plot_pies_calval(ax, color='k', marker='*'):
    """
    Plot Magdalena PIES locations P1–P5 on the axis.
    """
    for name in list(pies_calval.keys())[:5]:
        lat, lon = pies_calval[name]
        ax.plot(lon, lat, marker=marker, color=color)


def plot_pioneer2(ax, color='k', marker='v'):
    """
    Plot Pioneer II arrow mooring locations on the axis.
    """
    for name, (lat, lon, _) in pioneer.items():
        ax.plot(lon, lat, marker=marker, color=color)


def calval_extract_section_xoak(ds, sta1, sta2, spacing_km=1):

    # Generate spacing_km–spaced section
    lon_line, lat_line = equal_distance_line(sta1,        # starting point (lon, lat)
                                             sta2,        # end point (lon, lat)
                                             spacing_km   # spacing in km
                                            )

    # Create xarray for line (needed for xoak)
    section = xr.Dataset({
        "longitude": xr.DataArray(lon_line, dims=("station",)),
        "latitude": xr.DataArray(lat_line, dims=("station",)),
    })

    # ---------------------------------------------------------
    # stack num_lines and num_pixels locations
    dummy = ds.stack(coord=['num_lines', 'num_pixels'])
    print(dummy.sizes['coord'])

    # drop all values where not valid data
    dummy = dummy.where(dummy['quality_flag'] == 0, drop=True)
    print(dummy.sizes['coord'])

    # remove points with <90% temporal coverage
    dummy = dummy.where(
        dummy['ssha_filtered'].count(dim='time_cycle') / dummy.sizes['time_cycle'] > 0.9,
        drop=True
    )
    print(dummy.sizes['coord'])

    # use xoak to select stations
    dummy.xoak.set_index(["longitude", "latitude"], 'sklearn_geo_balltree')
    sec = dummy.xoak.sel(
        latitude=section.latitude,
        longitude=section.longitude
    )

    return sec,section
    


# extract ice shelves for plotting
from cartopy.feature import NaturalEarthFeature
ice_shelves = NaturalEarthFeature(
    category='physical',
    name='antarctic_ice_shelves_polys',
    scale='10m',
    facecolor='lightblue',
    edgecolor='black'
)


# ------------------------------------------------------------
# helper function to reduce complexity of subplots functions
# ------------------------------------------------------------
def plot_add(ax,levels = [50,75,100,200,1000]):
    ax.contour(bathy.x, bathy.y, bathy.z * (-1),
                levels=levels, colors='k',
                transform=ccrs.PlateCarree(),
                linewidths=1, alpha=0.3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                       color='lightgray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.add_feature(cfeature.GSHHSFeature(scale='high'), facecolor='lightgray', edgecolor='None')

# ------------------------------------------------------------
# Helper: convert a Matplotlib figure to an RGB image array
# ------------------------------------------------------------
def fig_to_rgb_array(fig):
    fig.set_dpi(200)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)



# ---------------------------------------------------------------------
# Any additional project-dependent imports can be appended here
# ---------------------------------------------------------------------
# sys.path.append(config['paths']['local_script'])
