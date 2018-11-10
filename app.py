from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pvlib
import rdtools
import matplotlib
import yaml
import requests
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# Load in config.yaml file
stream = open('config.yml', 'r')
config = yaml.load(stream)


def get_residential_utility():
    utility = requests.get('https://developer.nrel.gov/api/utility_rates/v3.json',
                        params={'api_key': config['api_key'],
                                'lat': config['latitude'],
                                'lon': config['longitude']})

    try:
        residential_price = utility.json()['outputs']['residential']
        if (type(residential_price) is str):
            print('No Utility data Available')
            residential_price = 0.1
    except Exception as e:
        print('Utility API Failed: Error Code {}'.format(utility.status_code))
        print(e)
        print()
        print('Using $0.10 / kWhr')
        residential_price = 0.1

    return residential_price


def total_discounting(t, degradation):
    present_value_discount = config['present_value_discounting']
    return ((1 + degradation)*present_value_discount)**(t / 365)


#utility = re.sub("[^0-9|.]", "", config['utility_bill'])


def predicted_output(df, t):
    base_efficiency = config['base_efficiency']
    past_year = np.array(pd.read_csv(config['past_year'], skiprows=4))
    day = df.index[-1].month*30 + df.index[-1].day
    return past_year[(day + t) % 365]*base_efficiency
    #utility = re.sub("[^0-9|.]", "", config['utility_bill'])


def full_output():
    df, rd = average_degradation()
    t = np.array(range(0, (20*365)))

    end = (1 + rd)**((df.index[-1] - df.index[0])
                                     .total_seconds() / (3600.0*24*365))

    better_output = predicted_output(df, t)

    normal_output = predicted_output(df, t)*end

    replace_profits = (-config['cost_of_new_solar'] +
                     (better_output*total_discounting(t, rd)*
                      get_residential_utility()).sum())

    normal_profits = normal_output*total_discounting(t, rd)*get_residential_utility()

    return (normal_profits.sum(), replace_profits)



def estimate_base(df, degradation):
    length = len(df)
    start = length // 2 - length // 10
    end = length // 2 + length // 10
    time_delta = pd.to_timedelta(df.power.index.freq).total_seconds()/(3600.0)

    average_middle = df.energy[start:end].mean()
    to_edges = time_delta*(end - length//2) / (24*365)

    start_energy = average_middle*(1 - degradation)**to_edges
    end_energy = average_middle*(1 + degradation)**to_edges

    return (start_energy, end_energy)


def average_degradation():
    file_name = config['real_time_data']

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

    # yoy_rd is mean degradation rate per year.
    return (df, 0.01*yoy_rd)

print(full_output())

# msg = MIMEMultipart('alternative')
# msg['Subject'] = "Daily Report"
# msg['From'] = me
# msg['To'] = you

# html = """\
# <html>
#   <head>
#   </head>
#   <body>
#     <div align="center"><h2>{0}</h2>
#     {1}
#     </div>
#     {2}
#     {3}
#     {4}
#     {5}
#     {6}
#     {7}
#     {8}
#     {9}
#   </body>
# </html>
# """.format(datestring,school_html,forecast_html,apod_html,xkcd_html,smbc_html,buttersafe_html,PDL_html,EC_html,dino_html)

# part1 = MIMEText(html, 'plain')
# part2= MIMEText(html, 'html')

# msg.attach(part1)
# msg.attach(part2)

# mail = smtplib.SMTP('smtp.gmail.com', 587)
# mail.ehlo()
# mail.starttls()
# mail.login(me, 'josephjoseph')
# mail.sendmail(me,you,msg.as_string())
# m
