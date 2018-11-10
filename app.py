from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pvlib
import rdtools
import matplotlib
import yaml
import requests

# Load in config.yaml file
stream = open('config.yml', 'r')
config = yaml.load(stream)
file_name = config['real_time_data']

utility = requests.get('https://developer.nrel.gov/api/utility_rates/v3.json',
                       params={'api_key': config['api_key'],
                               'lat': config['latitude'],
                               'lon': config['longitude']})

try:
    residential_price = utility.json()['outputs']['residential']
except requests.RequestException as e:
    print('Utility API Failed: Error Code {}'.format(utility.status_code))
    print(e)
    print()
    print('Using $0.10 / kWhr')
    residential_price = 0.1

# pd.plotting.register_matplotlib_converters()
# matplotlib.rcParams.update({'font.size': 12,
#                            'figure.figsize': [4.5, 3],
#                            'lines.markeredgewidth': 0,
#                            'lines.markersize': 2
#                            })

df = pd.read_csv(file_name)
df = df.rename(columns = {
    u'12 BP Solar - Active Power (kW)':'power',
    u'12 BP Solar - Wind Speed (m/s)': 'wind',
    u'12 BP Solar - Weather Temperature Celsius (\xb0C)': 'Tamb',
    u'12 BP Solar - Global Horizontal Radiation (W/m\xb2)': 'ghi',
    u'12 BP Solar - Diffuse Horizontal Radiation (W/m\xb2)': 'dhi'
})

# Specify the Metadata
meta = config

df.index = pd.to_datetime(df.Timestamp)
# TZ is required for irradiance transposition
df.index = df.index.tz_localize(meta['timezone'], ambiguous = 'infer')

# Explicitly trim the dates so that runs of this example notebook
# are comparable when the source dataset has been downloaded at different times
df = df[config['start_date']:config['end_date']]

# Change power from kilowatts to watts
df['power'] = df.power * 1000.0
# There is some missing data, but we can infer the frequency from the first several data points
freq = pd.infer_freq(df.index[:10])

# And then set the frequency of the dataframe
df = df.resample(freq).median()

# Calculate energy yield in Wh
df['energy'] = df.power * pd.to_timedelta(df.power.index.freq).total_seconds()/(3600.0)

# Calculate POA irradiance from DHI, GHI inputs
loc = pvlib.location.Location(meta['latitude'], meta['longitude'], tz = meta['timezone'])
sun = loc.get_solarposition(df.index)

# calculate the POA irradiance
sky = pvlib.irradiance.isotropic(meta['tilt'], df.dhi)
df['dni'] = (df.ghi - df.dhi)/np.cos(np.deg2rad(sun.zenith))
beam = pvlib.irradiance.beam_component(meta['tilt'], meta['azimuth'], sun.zenith, sun.azimuth, df.dni)
df['poa'] = beam + sky

# Calculate cell temperature
df_temp = pvlib.pvsystem.sapm_celltemp(df.poa, df.wind, df.Tamb, model = meta['temp_model'])
df['Tcell'] = df_temp.temp_cell

# plot the AC power time series
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(df.index, df.power, 'o', alpha = 0.05)
ax.set_ylim(0,7000)
fig.autofmt_xdate()
ax.set_ylabel('AC Power (W)')

# Specify the keywords for the pvwatts model
pvwatts_kws = {"poa_global" : df.poa,
              "P_ref" : meta['pdc'],
              "T_cell" : df.Tcell,
              "G_ref" : 1000,  # reference irradiance
              "T_ref": 25,  # reference temperature
              "gamma_pdc" : meta['tempco']}

# Calculate the normaliztion, the function also returns the relevant insolation for
# each point in the normalized PV energy timeseries
normalized, insolation = rdtools.normalize_with_pvwatts(df.energy, pvwatts_kws)

df['normalized'] = normalized
df['insolation'] = insolation

# Calculate a collection of boolean masks that can be used
# to filter the time series
nz_mask = (df['normalized'] > 0)
poa_mask = rdtools.poa_filter(df['poa'])
tcell_mask = rdtools.tcell_filter(df['Tcell'])
clip_mask = rdtools.clip_filter(df['power'])

# filter the time series and keep only the columns needed for the
# remaining steps
filtered = df[nz_mask & poa_mask & tcell_mask & clip_mask]
filtered = filtered[['insolation', 'normalized']]

daily = rdtools.aggregation_insol(filtered.normalized, filtered.insolation, frequency = 'D')

# Calculate the degradation rate using the YoY method
yoy_rd, yoy_ci, yoy_info = rdtools.degradation_year_on_year(daily, confidence_level=68.2)
# Note the default confidence_level of 68.2 is approrpriate if you would like to
# report a confidence interval analogous to the standard deviation of a normal
# distribution. The size of the confidence interval is adjustable by setting the
# confidence_level variable.

# Visualize the results
start = daily.index[0]
end = daily.index[-1]
years = (end - start).days / 365.0
yoy_values = yoy_info['YoY_values']

x = [start, end]
y = [1, 1 + (yoy_rd * years)/100]
