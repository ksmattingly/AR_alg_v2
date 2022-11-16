# AR algorithm v2

Version 2 of Mattingly atmospheric river (AR) identification algorithm. The algorithm identifies AR outlines by applying size, shape, and transport direction criteria to fields of integrated water vapor transport (IVT) from gridded meteorological analyses.

Version 1 of the algorithm is described in [Mattingly et al. (2018)](https://doi.org/10.1029/2018JD028714). The AR identification criteria are based on the [Guan and Waliser (2015)](https://doi.org/10.1002/2015JD024257) and [Mundhenk et al. (2016)](https://doi.org/10.1175/JCLI-D-15-0655.1) algorithms, but with less strict requirements for minimum IVT threshold, object shape, and transport direction in order to capture ARs in the polar regions. This version (v2) includes the following updates:
- Reorganization of the code for speed and clarity
- Option to easily change AR identification parameters through the included configuration file
- Ability to calculate IVT on native model levels in addition to constant pressure levels
- Ability to identify ARs in the Southern Hemisphere as well as the Northern Hemisphere
- Default behavior is to use IVT vector components for transport direction, rather than lower tropospheric mean wind vector
- Default cutoff latitude for AR meridional transport requirement is now the Arctic / Antarctic circle (66.56&deg;N / -66.56&deg;S), rather than 70&deg;N / -70&deg;S

To date, the algorithm has been successfully used to identify ARs in MERRA-2 and ERA5 reanalysis data. Please feel free to [contact me](mailto:ksmattingly@wisc.edu) if you are interested in applying the algorithm to other gridded meteorological data sources.

## Dependencies

The code requires a python 3 environment with the following packages:
- numpy
- scipy
- xarray
- pandas
- netcdf4
- scikit-image
- geopy
- hjson

Additionally, the directory containing the project code must be included in the user's python path.

## How to run

The AR identification procedure requires gridded fields of IVT magnitude, u- and v-IVT components, and IVT values at the desired climatological percentile rank threshold (default: 85th percentile) as input. Optionally, lower-tropospheric mean wind may be used in place of u- and v-IVT components for transport direction criteria. The code is somewhat flexible with regard to the time properties of input and output files (the start time, end time, and interval between timesteps in hours); for data with 6-hourly or more frequent temporal resolution, structuring the data as monthly files is recommended.

The typical work flow for producing AR data is:
- Edit configuration options in `AR_ID_config.hjson`. These options including the input data source, data directories, input file characteristics, spatial domain and grid, IVT calculation parameters, and AR identification parameters. More details on each item can be found in the comments in the configuration file.
- Run `calc_IVT.py <begin_time> <end_time> <timestep_hrs>` to calculate IVT u/v-components and vector magnitude from 3D input fields of u- & v-wind components and specific humidity.
  - Use `calc_IVT_ERA5.py` to calculate IVT magnitude from the u- and v-IVT fields provided as pre-calculated fields in ERA5.
- Run `calc_IVT_at_percentiles.py <start_year> <end_year> <timestep_hrs> <start_doy> <end_doy>` to calculate IVT values at the climatological percentile rank(s) specified in the configuration file (with climatology defined as the 31-day centered IVT distribution for the specified `start_year`, `end_year`, and `timestep_hrs`).
- Run `ARs_ID.py <begin_time> <end_time> <timestep_hrs>` to identify final AR outlines.

Two important notes on running `ARs_ID.py`:
- The AR identification code must be run separately for the northern and southern hemisphere. Minimum (maximum) latitude of NH (SH) AR output data is 10&deg;N (-10&deg;S). However, global IVT and IVT at percentiles files can be used as input to the AR algorithm.
- The input IVT and IVT at percentiles files must span the entire globe zonally (e.g. MERRA-2 data must have longitudes extending from -180&deg;W to 179.375&deg;E).

## Contact

Kyle Mattingly
Space Science and Engineering Center
University of Wisconsin-Madison
[ksmattingly@wisc.edu](mailto:ksmattingly@wisc.edu)