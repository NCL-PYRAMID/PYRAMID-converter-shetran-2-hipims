###############################################################################
# Converter - SHETRAN to HiPIMS
# Xue Tong, Amy Green, Robin Wardle
# July 2023
###############################################################################

###############################################################################
# Python libraries
###############################################################################
import imp
import os
import pathlib
import datetime
import shutil
import fiona
import rasterio
import rasterio.mask
import rasterio.fill
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d
import pandas as pd
from os.path import join
import sys
import logging
import subprocess


###############################################################################
# Constants
###############################################################################
CONVERTER_SUCCESS_FILENAME = "success"
CONVERTER_LOG_FILENAME = "converter-s2h.log"
METADATA_FILENAME = "metadata.json"


###############################################################################
# Paths
###############################################################################
# Setup base path
platform = os.getenv("PLATFORM")
if platform=="docker":
    data_path = os.getenv("DATA_PATH", "/data")
else:
    data_path = os.getenv("DATA_PATH", "./data")

# INPUT data paths and files
input_path = data_path / pathlib.Path("inputs")
bound_line = input_path / pathlib.Path("Line2.shp")
SHETRAN_cells = input_path / pathlib.Path("Tyne_at_Newcastle_DEM.asc")
shetran_h5 = input_path / pathlib.Path("output_Tyne_at_Newcastle_shegraph.h5")

# OUTPUT data paths and files
output_path = data_path / pathlib.Path("outputs")
    
# Remove the output path if it exists, and create a new one
if output_path.exists() and output_path.is_dir():
    shutil.rmtree(output_path)
pathlib.Path.mkdir(output_path)

hipims_outpath = join(output_path, "HIPIMS")
if not os.path.exists(hipims_outpath):
    os.mkdir(hipims_outpath)


###############################################################################
# Logging
###############################################################################
# Configure logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)

# Logging instance
logger = logging.getLogger(pathlib.PurePath(__file__).name)
logger.propagate = False

# Console messaging
console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File logging
file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(output_path / pathlib.Path(CONVERTER_LOG_FILENAME))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info("Logger initialised")

# Some additional logging info
logger.info("DATA_PATH = {}".format(data_path))
logger.info("output_path = {}".format(output_path))


###############################################################################
# Environmental Parameters
#   Parameters are passed as environment variables,
#   i.e. they are strings and will need to be converted where necessary
###############################################################################
# INPUT data paramters (need to change to global for DAFNI)
try:
    start_datetime = pd.to_datetime(os.getenv("RUN_START_DATE", "2023-06-20 12:00:00"), utc=True)
    duration = float(os.getenv("HIPIMS_RUN_DURATION", 6.0)) # hours
    # Read in the river cell to extract SHETRAN flows from:
    river_cell = int(os.getenv("SHETRAN_RIVER_CELL", 484)) # 484 is the Tyne near Tyne Bridge
except (TypeError, ValueError, Exception) as e:
    logger.error("Error converting parameter ", exc_info=e)
    raise
end_datetime = start_datetime + pd.Timedelta(duration, "h")

print("\033[0;31;40m---> start_datetime: {}\033[0;37;40m".format(start_datetime))
print("\033[0;31;40m---> end_datetime: {}\033[0;37;40m".format(end_datetime))


###############################################################################
# Start Conversion
###############################################################################
f_inflows = "Shetran_bound.txt"
f_mask = "She2HiMask.tif"

# read Hipims simulation boundary 
with fiona.open(input_path / bound_line,"r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

# read SHETRAN cells to extract cell numbers overlapped
with rasterio.open(input_path / SHETRAN_cells) as src:
    # read DEM and get DEM boundary
    demMasked = src.read(1, masked=True)
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=False)  
    dem = np.ma.filled(demMasked, fill_value=-9999.)
    out_meta = src.meta

    # mask boundary
    mask_boundary = np.zeros_like(dem)
    mask_boundary[out_image[0,:,:] > 0] = 1

    row, col = np.shape(mask_boundary)
    x = np.zeros_like(dem)
    y = np.zeros_like(dem)
    for i in range(col):
        y[:, i] = i
    for i in range(row):
        x[i, :] = i
    
    x_bound = x[mask_boundary > 0]
    y_bound = y[mask_boundary > 0]

# get rainfall data for hipims
rainfall_data_path = join(join(input_path, "HIPIMS"), "rain_source.txt")
rainfall = pd.read_csv(rainfall_data_path, index_col=0)
rainfall.index = pd.to_datetime(rainfall.index, utc=True)
hipims_rainfall = rainfall.loc[start_datetime : end_datetime]
hipims_rainfall = hipims_rainfall / 1000 /3600  # Convert from mm/hr to m/s
rainfall_timestep_secs = (hipims_rainfall.index[1] - hipims_rainfall.index[0]).seconds # duration of rainfall timestep in  seconds

times2 = (np.arange(len(hipims_rainfall)) * 15 * 60).astype(int) # 15 minute timesteps for HiPIMS rainfall inputs, duplicated below in hipims_timesteps.
# TODO - change this so that the timesteps are taken from the input data, so that it doesn't always have to be 15 minutes.

# save rainfall data in correct format for HIPIMS
hipims_rainfall_outpath = join(output_path, "HIPIMS")
os.makedirs(hipims_rainfall_outpath, exist_ok=True)
hipims_rainfall_output = pd.DataFrame(hipims_rainfall.to_numpy(), index=times2)
hipims_rainfall_output.to_csv(join(hipims_rainfall_outpath, "rain_source.txt"), header=False, sep = " ")

# generate new mask
maskId = np.zeros_like(x) - 9999
bound_count = np.size(x_bound)
for i in range(bound_count):
    xi = int(x_bound[i])
    yi = int(y_bound[i])
    maskId[xi, yi] = i
with rasterio.open(output_path / f_mask, "w", **out_meta) as dest:
    dest.write(maskId,1)

logger.info("mask generated!")

### extract corresponding discharge from SHETRAN 

with h5py.File(input_path / shetran_h5, 'r', driver='core') as hf:
    f_keys = hf["VARIABLES"].keys()
    f_key = [k for k in f_keys if 'ovr_flow' in k]
    discharge = hf["VARIABLES"][f_key[0]]['value'][:]  # The variable is not always called "  1 ovr_flow", this will find it, else you can specify directly.
# TODO: ensure that this works with other visualisation plan inputs (i.e. when ovr_flow for grids, not rivers, is included).
print("\033[0;31;40m---> discharge: \n{}\033[0;37;40m".format(discharge))

# Find which direction has the greatest flow - this is the orthagonal downstream flow. 
direction = np.argmax(abs(np.sum(discharge[river_cell, :, 0:1000], axis=1)))

# Extract the SHETRAN flow from the H5 file.
# The river cell was taken manually from the Element Numbering variable in the H5, but should be automated in the future.
discharge = discharge[river_cell, direction, :]  # Dimentions [cell no., N/E/S/W, time]

logger.info('h5 read!')

# Create Pandas Dataframes with Dates:
# shetran_startdate = "1989-12-31" # I (Amy) THINK THIS IS WRONG AND NEEDS CHANGING TO START DATE OF RAIN_SOURCE.TXT 
shetran_startdate = rainfall.index[0]  # Format: "1990-01-04 00:00:00"
print("\033[0;31;40m---> shetran_startdate: {}\033[0;37;40m".format(shetran_startdate))

flows = pd.DataFrame(data={'flow': discharge},
                     index=pd.date_range(shetran_startdate,
                                         periods=len(discharge),
                                         freq="H"))
source = flows["flow"].loc[start_datetime : end_datetime]

if len(source)==0:
    text = f'No flows were extracted from SHETRAN! This is likely to do with the date parameters. shetran_startdate ({shetran_startdate}) should be within the start ({start_datetime}) and end ({end_datetime}) dates.'
    print(text)
    logger.info(text) 

# interpolate discharge time series
source_duration_secs = (source.index[-1] - source.index[0]).seconds  # Duration the source data covers
shetran_timestep_secs = (source.index[1] - source.index[0]).seconds  # Duration of each source timestep

# Then create some new timesteps for interpolating from/to:
shetran_timesteps = np.arange(0, (source_duration_secs + shetran_timestep_secs), shetran_timestep_secs) # hourly discharge (seconds)
hipims_timesteps = np.arange(0, ((len(source)-1)*shetran_timestep_secs)+rainfall_timestep_secs, rainfall_timestep_secs)  # 15 minute discharge (seconds)
# TODO: check whether you need hipims_timesteps if you already have times2 above
# times1 = np.arange(duration) * 3600 # hourly discharge

#print("\033[0;31;40m---> source: \n{}\033[0;37;40m".format(source))
#print("\033[0;31;40m---> discharge1: \n{}\033[0;37;40m".format(discharge1))
#print("\033[0;31;40m---> len(times1): {}\033[0;37;40m".format(len(times1)))
#print("\033[0;31;40m---> len(discharge1): {}\033[0;37;40m".format(len(discharge1)))

# print(shetran_timesteps, "---", hipims_timesteps, "---", source.values)
f = interp1d(shetran_timesteps, source.values, kind='cubic')

discharge2 = f(hipims_timesteps) / 1e6

# Stack the times and the shetran data:
Shetran_bound = np.vstack((hipims_timesteps, discharge2))

# Add on the final shetran time and data point (missed from interpolation):
# q_end = np.append([duration * 3600], [source[-1]], axis=0) # check this - Ben thinks this is just dur*3600, not len(dur)*3600.
# q_end = np.array([q_end])

# Merge some rows to produce a 2 column dataset of times and flows:
Shetran_bound = np.r_[Shetran_bound.T]  # , q_end]

# Add on a column of 0s - these represent flow in the y direction, the columns are time, flow in x, flow in y. For the Tyne, the flow is from the west (x).
Shetran_bound = np.append(Shetran_bound, np.zeros([len(Shetran_bound),1]), 1)

# Save the data
np.savetxt(output_path / f_inflows, Shetran_bound)

#completed = subprocess.run(["head", f_inflows], stdout=subprocess.PIPE, encoding="utf-8")
#print("\033[0;31;40m---> 'head {}': \n{}\033[0;37;40m".format(f_inflows, completed.stdout))
#completed = subprocess.run(["tail", f_inflows], stdout=subprocess.PIPE, encoding="utf-8")
#print("\033[0;31;40m---> 'tail {}': \n{}\033[0;37;40m".format(f_inflows, completed.stdout))

#print("\033[0;31;40m---> 'tail Shetran_bound': \n{}\033[0;37;40m".format(Shetran_bound))


logger.info("inflow text generated!")

# print("--- HiPIMS start")
# print(hipims_rainfall_output[0:5])
# print("--- HiPIMS end")
# print(hipims_rainfall_output[-5:1])
# print("--- SHETRAN source")
# print(source.values)
# print("--- SHETRAN start")
# print(Shetran_bound[0:10])
# print("--- SHETRAN end")
# print(Shetran_bound[-10:-1])

###############################################################################
# Metadata
###############################################################################
title = os.getenv('TITLE', 'SHETRAN-2-HiPIMS Simulation')
description = 'Convert outputs from SHETRAN to inputs for HiPIMS'
geojson = {}


metadata = f"""{{
  "@context": ["metadata-v1"],
  "@type": "dcat:Dataset",
  "dct:language": "en",
  "dct:title": "{title}",
  "dct:description": "{description}",
  "dcat:keyword": [
    "shetran"
  ],
  "dct:subject": "Environment",
  "dct:license": {{
    "@type": "LicenseDocument",
    "@id": "https://creativecommons.org/licences/by/4.0/",
    "rdfs:label": null
  }},
  "dct:creator": [{{"@type": "foaf:Organization"}}],
  "dcat:contactPoint": {{
    "@type": "vcard:Organization",
    "vcard:fn": "DAFNI",
    "vcard:hasEmail": "support@dafni.ac.uk"
  }},
  "dct:created": "{datetime.datetime.now().isoformat()}Z",
  "dct:PeriodOfTime": {{
    "type": "dct:PeriodOfTime",
    "time:hasBeginning": null,
    "time:hasEnd": null
  }},
  "dafni_version_note": "created",
  "dct:spatial": {{
    "@type": "dct:Location",
    "rdfs:label": null
  }},
  "geojson": {geojson}
}}
"""
with open(os.path.join(output_path, METADATA_FILENAME), 'w') as f:
    f.write(metadata)

# Finished successfully
logger.info("Model execution completed successfully")
pathlib.Path(os.path.join(output_path, CONVERTER_SUCCESS_FILENAME)).touch()
