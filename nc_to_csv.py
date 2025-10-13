import netCDF4
import pandas as pd
import os

# Replace with your NetCDF file path
filename = 'mvp_profile_stationary_conv'
netcdf_file = filename + '.nc'
csv_file = filename + '.csv'
netcdf_path = os.path.join('C:/Users/maxim/Documents/INGE CNRS/ESSTECH25/netcdf_fro_seabird/', netcdf_file)
output_path = os.path.join('C:/Users/maxim/Documents/INGE CNRS/ESSTECH25/csv_for_seabird/', csv_file)

ds = netCDF4.Dataset(netcdf_path)

# Collect variables that are 1D and can be columns
columns = {}
for var in ds.variables:
    data = ds.variables[var][:]
    if data.ndim == 1 and len(data) == ds.dimensions[list(ds.dimensions.keys())[0]].size:
        columns[var] = data[:]

# Create DataFrame and save to CSV
df = pd.DataFrame(columns)
df.to_csv(output_path, index=False)

ds.close()
print(f"CSV saved to {output_path}")