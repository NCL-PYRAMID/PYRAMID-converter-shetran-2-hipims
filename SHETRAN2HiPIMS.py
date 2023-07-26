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
    start_datetime = pd.to_datetime(os.getenv("RUN_START_DATE", "2012-06-28 12:00:00"), utc=True)
    duration = float(os.getenv("HIPIMS_RUN_DURATION", 6.0)) # hours
except (TypeError, ValueError, Exception) as e:
    logger.error("Error converting parameter ", exc_info=e)
    raise
end_datetime = start_datetime + pd.Timedelta(duration, "h")


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

times2 = (np.arange(len(hipims_rainfall)) * 60 ** 2 / 15).astype(int)

# save rainfall data in correct format for HIPIMS
hipims_rainfall_outpath = join(output_path, "HIPIMS")
os.makedirs(hipims_rainfall_outpath, exist_ok=True)
pd.DataFrame(hipims_rainfall.to_numpy(), index=times2).to_csv(join(hipims_rainfall_outpath, "rain_source.txt"), header=False)

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
    discharge = hf["VARIABLES"]["  1 ovr_flow"]["value"][:]
logger.info('h5 read!')
bound_count = np.size(x_bound)
source = []
# Create Pandas Dataframes with Dates:
#shetran_startdate = "1989-12-31" # I THINK THIS IS WRONG AND NEEDS CHANGING TO START DATE OF RAIN_SOURCE.TXT 
shetran_startdate = rainfall.index[0]

print("-----")
print(bound_count)
print(x_bound)
print("-----")
print(y_bound)
print("-----")

for i in range(bound_count):
    x = int(x_bound[i])
    y = int(y_bound[i])
    flows_N = pd.DataFrame(data={'flow': discharge[x+1,y+1,0,0,:]},
                       index=pd.date_range(shetran_startdate,
                       periods=len(discharge[x+1,y+1,0,0,:]),
                       freq="H"))
    flows_E = pd.DataFrame(data={'flow': discharge[x+1,y+1,0,1,:]},
                       index=pd.date_range(shetran_startdate,
                       periods=len(discharge[x+1,y+1,0,0,:]),
                       freq="H"))
    flows_S = pd.DataFrame(data={'flow': discharge[x+1,y+1,0,2,:]},
                       index=pd.date_range(shetran_startdate,
                       periods=len(discharge[x+1,y+1,0,0,:]),
                       freq="H"))
    flows_W = pd.DataFrame(data={'flow': discharge[x+1,y+1,0,3,:]},
                       index=pd.date_range(shetran_startdate,
                       periods=len(discharge[x+1,y+1,0,0,:]),
                       freq="H"))
    source_N = flows_N["flow"].loc[start_datetime : end_datetime]
    source_E = flows_E["flow"].loc[start_datetime : end_datetime]
    source_S = flows_S["flow"].loc[start_datetime : end_datetime]
    source_W = flows_W["flow"].loc[start_datetime : end_datetime]
    sourcei = source_S - source_N + source_W - source_E
    source.append(sourcei)

# interpolate discharge time series

times1 = np.arange(duration) * 3600 # hourly discharge

shetran_sourcei = []
q_end = []
for i in range(bound_count):
    discharge1 = source[i][:]
    f = interp1d(times1, discharge1, kind='cubic')
    discharge2 = f(times2) / 1e6
    q_end.append(source[i][-1])
    shetran_sourcei.append(discharge2)

Shetran_bound = np.vstack((times2, shetran_sourcei))
q_end = np.append([len(duration) * 3600], q_end, axis=0) # check this
q_end = np.array([q_end])

Shetran_bound = np.r_[Shetran_bound.T, q_end]
np.savetxt(output_path / f_inflows, Shetran_bound)
logger.info("inflow text generated!")


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
