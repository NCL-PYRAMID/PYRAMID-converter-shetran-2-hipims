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

###############################################################################
# Paths
###############################################################################
# Setup base path
platform = os.getenv("PLATFORM")
if platform=="docker":
    data_path = os.getenv("DATA_PATH", "/data")
else:
    data_path = os.getenv("DATA_PATH", "/Users/cusnow/Job/SHETRAN2Hipims/SHETRAN2Hipims/data")

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
    mask_boundary[out_image[0,:,:]>0] = 1

    row, col = np.shape(mask_boundary)
    x = np.zeros_like(dem)
    y = np.zeros_like(dem)
    for i in range(col):
        y[:, i] = i
    for i in range(row):
        x[i, :] = i
    
    x_bound = x[mask_boundary > 0]
    y_bound = y[mask_boundary > 0]

# generate new mask
maskId = np.zeros_like(x) - 9999
bound_count = np.size(x_bound)
for i in range(bound_count):
    xi = int(x_bound[i])
    yi = int(y_bound[i])
    maskId[xi, yi] = i
with rasterio.open(output_path / f_mask, "w", **out_meta) as dest:
    dest.write(maskId,1)

print("mask generated!")

### extract corresponding discharge from SHETRAN 
with h5py.File(input_path / shetran_h5, 'r', driver='core') as hf:
    discharge = hf["VARIABLES"]["  1 ovr_flow"]["value"][:] 
print('h5 read!')
bound_count = np.size(x_bound)
source = []
# Create Pandas Dataframes with Dates:
for i in range(bound_count):
    x = int(x_bound[i])
    y = int(y_bound[i])
    flows_N = pd.DataFrame(data={'flow': discharge[x+1,y+1,0,0,:]},
                       index=pd.date_range("1989-12-31", 
                       periods=len(discharge[x+1,y+1,0,0,:]), 
                       freq="H"))
    flows_E = pd.DataFrame(data={'flow': discharge[x+1,y+1,0,1,:]},
                       index=pd.date_range("1989-12-31", 
                       periods=len(discharge[x+1,y+1,0,0,:]), 
                       freq="H"))
    flows_S = pd.DataFrame(data={'flow': discharge[x+1,y+1,0,2,:]},
                       index=pd.date_range("1989-12-31", 
                       periods=len(discharge[x+1,y+1,0,0,:]), 
                       freq="H"))
    flows_W = pd.DataFrame(data={'flow': discharge[x+1,y+1,0,3,:]},
                       index=pd.date_range("1989-12-31", 
                       periods=len(discharge[x+1,y+1,0,0,:]), 
                       freq="H"))
    source_N = flows_N["flow"].loc["2012-06-28 12:00:00":"2012-06-28 20:00:00"]
    source_E = flows_E["flow"].loc["2012-06-28 12:00:00":"2012-06-28 20:00:00"]
    source_S = flows_S["flow"].loc["2012-06-28 12:00:00":"2012-06-28 20:00:00"]
    source_W = flows_W["flow"].loc["2012-06-28 12:00:00":"2012-06-28 20:00:00"]
    sourcei = source_S - source_N + source_W - source_E
    source.append(sourcei)

# interpolate discharge time serie
times1 = np.arange(9) * 3600
times2 = np.arange(8 * 6) * 600
shetran_sourcei = []
q_end = []
for i in range(bound_count):
    discharge1 = source[i][:]
    f = interp1d(times1, discharge1, kind='cubic')
    discharge2 = f(times2) / 1e6
    q_end.append(source[i][-1])
    shetran_sourcei.append(discharge2)

Shetran_bound = np.vstack((times2, shetran_sourcei))
q_end = np.append([8*3600], q_end, axis=0)
q_end = np.array([q_end])

Shetran_bound = np.r_[Shetran_bound.T, q_end]
np.savetxt(output_path / f_inflows, Shetran_bound)
print("inflow text generated!")

title = os.getenv('TITLE', 'SHETRAN-2-hipims Simualtion')
description = 'Convert outputs from SHETRAN to inputs for HiPIMS.'
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
with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
    f.write(metadata)