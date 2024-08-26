import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import yaml
import numpy as np
from TSI_el_az import el_az_changer, TSI
import joblib
import json

# Load YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#period of prediction and sensor locations from the config
period_of_prediction = config['period_of_prediction']
sensor_locations = config['sensors']

# Setup the Open-Meteo API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


#function to fetch forecast data
def fetch_forecast_data_for_location(lat, lon, forecast_days):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation",
                   "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
                   "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration",
                   "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
                   "wind_direction_100m", "wind_gusts_10m", "shortwave_radiation", "direct_radiation",
                   "diffuse_radiation", "global_tilted_irradiance", "direct_normal_irradiance",
                   "terrestrial_radiation"],
        "wind_speed_unit": "ms",
        "timezone": "auto",
        "forecast_days": forecast_days
    }
    response = openmeteo.weather_api(url, params=params)
    return response[0]


# Fetch forecast data for each sensor location
weather_data_dict = {}

for sensor, properties in sensor_locations.items():
    lat, lon = properties['coordinates']
    print(f"Fetching forecast data for {sensor} at ({lat}, {lon})...")
    response = fetch_forecast_data_for_location(lat, lon, period_of_prediction)

    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["dew_point_2m"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["apparent_temperature"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["precipitation"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["rain"] = hourly.Variables(5).ValuesAsNumpy()
    hourly_data["snowfall"] = hourly.Variables(6).ValuesAsNumpy()
    hourly_data["snow_depth"] = hourly.Variables(7).ValuesAsNumpy()
    hourly_data["weather_code"] = hourly.Variables(8).ValuesAsNumpy()
    hourly_data["pressure_msl"] = hourly.Variables(9).ValuesAsNumpy()
    hourly_data["surface_pressure"] = hourly.Variables(10).ValuesAsNumpy()
    hourly_data["cloud_cover"] = hourly.Variables(11).ValuesAsNumpy()
    hourly_data["cloud_cover_low"] = hourly.Variables(12).ValuesAsNumpy()
    hourly_data["cloud_cover_mid"] = hourly.Variables(13).ValuesAsNumpy()
    hourly_data["cloud_cover_high"] = hourly.Variables(14).ValuesAsNumpy()
    hourly_data["et0_fao_evapotranspiration"] = hourly.Variables(15).ValuesAsNumpy()
    hourly_data["vapour_pressure_deficit"] = hourly.Variables(16).ValuesAsNumpy()
    hourly_data["wind_speed_10m"] = hourly.Variables(17).ValuesAsNumpy()
    hourly_data["wind_speed_100m"] = hourly.Variables(18).ValuesAsNumpy()
    hourly_data["wind_direction_10m"] = hourly.Variables(19).ValuesAsNumpy()
    hourly_data["wind_direction_100m"] = hourly.Variables(20).ValuesAsNumpy()
    hourly_data["wind_gusts_10m"] = hourly.Variables(21).ValuesAsNumpy()
    hourly_data["shortwave_radiation"] = hourly.Variables(22).ValuesAsNumpy()
    hourly_data["direct_radiation"] = hourly.Variables(23).ValuesAsNumpy()
    hourly_data["diffuse_radiation"] = hourly.Variables(24).ValuesAsNumpy()
    hourly_data["global_tilted_irradiance"] = hourly.Variables(25).ValuesAsNumpy()
    hourly_data["direct_normal_irradiance"] = hourly.Variables(26).ValuesAsNumpy()
    hourly_data["terrestrial_radiation"] = hourly.Variables(27).ValuesAsNumpy()

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    weather_data_dict[sensor] = hourly_dataframe

# Iterate over each DataFrame in the dictionary
for sensor, df in weather_data_dict.items():
    print(f"Processing data for {sensor}...")

    if 'date' in df.columns:
        # Convert the 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Remove the timezone information
        df['date'] = df['date'].dt.tz_localize(None)

        # Set the 'date' column as the index
        df.set_index('date', inplace=True)

        # Update the DataFrame in the dictionary
        weather_data_dict[sensor] = df

        # Display the DataFrame info
        # print(weather_data_dict[sensor].info())
    else:
        print(f"Error: 'date' column not found in the data for {sensor}")


# Function to compute TSI, Azimuth, and Elevation
def compute_tsi_el_az(df, lat, lon):
    df['TSI'] = df.index.to_series().apply(TSI)
    df[['Azimuth', 'Elevation']] = df.index.to_series().apply(lambda dt: pd.Series(el_az_changer(dt, lat=lat, lte=lon)))
    return df


# Function to add time-based and trigonometric features
def add_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear

    df['Hour sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['Hour cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['Month sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['Month cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df


# Load the saved model
model_filename = r'C:\Users\safaa.lahnine\PycharmProjects\Eatemp\PredictionEatempModel.joblib'
loaded_model = joblib.load(model_filename)


hottest_months = [6, 7, 8]



def adjust_predictions(predictions, dates):
    adjusted_predictions = []
    for temp, date in zip(predictions, dates):
        if date.month in hottest_months:
            if 30 <= temp < 40:
                adjusted_temp = temp + 4.90
            elif temp >= 40:
                adjusted_temp = temp + 8
            else:
                adjusted_temp = temp
        else:
            adjusted_temp = temp
        adjusted_predictions.append(adjusted_temp)
    return np.array(adjusted_predictions)


# Initialize a dictionary to store predictions for all sensors
predictions_dict = {}

# Iterate over each sensor's weather data
for sensor, weather_df in weather_data_dict.items():
    lat, lon = sensor_locations[sensor]['coordinates']
    weather_df = compute_tsi_el_az(weather_df, lat, lon)

    # Calculate the cumulative sum of direct normal irradiance for each day
    features = [
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "global_tilted_irradiance",
        "direct_normal_irradiance",
        "terrestrial_radiation",
        "TSI"
    ]
    for feature in features:
        weather_df[f'cumulative_{feature}'] = weather_df[feature].groupby(weather_df.index.date).transform(
            pd.Series.cumsum)

    # Add time-based and trigonometric features
    weather_df = add_time_features(weather_df)

    # Create lagged features
    lag_features = ['apparent_temperature']
    for feature in lag_features:
        for lag in range(1, 25):  # Create lag features for the past 24 hours
            weather_df[f'{feature}_lag_{lag}'] = weather_df[feature].shift(lag)

    # Create rolling averages
    rolling_window = 96  # 24-hour rolling window for 15-minute intervals
    for feature in lag_features:
        weather_df[f'{feature}_rolling_mean'] = weather_df[feature].rolling(window=rolling_window).mean()

    # Drop rows with NaN values generated by shifting and rolling operations
    weather_df.dropna(inplace=True)

    # Filter the necessary columns for the prediction model
    weather_df['apparent_temperature'] = weather_df['apparent_temperature'].round(1)
    best_list = [
        'apparent_temperature', 'rain', 'wind_speed_10m', 'cloud_cover', 'relative_humidity_2m',
        'Azimuth', 'Elevation', 'global_tilted_irradiance', 'cumulative_global_tilted_irradiance',
        'shortwave_radiation', 'cumulative_shortwave_radiation', 'Hour sin', 'Hour cos',
        'Month sin', 'Month cos'
    ]
    weather_df = weather_df.filter(items=best_list)
    weather_df['cloud_cover'] = weather_df['cloud_cover'] / 100.0
    weather_df['relative_humidity_2m'] = weather_df['relative_humidity_2m'] / 100.0
    print(weather_df.head())

    # Make predictions
    weather_df['rail_temp_pred'] = loaded_model.predict(weather_df)

    # Adjust predictions
    weather_df['rail_temp_pred'] = adjust_predictions(weather_df['rail_temp_pred'], weather_df.index)

    # Store the predictions for this sensor
    predictions_df = weather_df[['rail_temp_pred']].copy()
    predictions_df['datetime'] = predictions_df.index
    predictions_dict[sensor] = predictions_df
    print(predictions_df.describe())

import struct
import json
import requests
import time


# function create outMsg
def creer_outmsg(deveui, numero_serie, timestamp, temperature):
    # Ensure the serial number has 4 digits
    numero_serie = f"{int(numero_serie):04d}"

    # Format the temperature
    temperature_value = int(round(temperature * 10))
    temperature_hex = struct.pack('<H', temperature_value).hex()

    #timestamp in little-endian format
    timestamp_hex = struct.pack('<I', int(timestamp)).hex().upper()

    # Assemble the payload_cleartext
    payload_cleartext = f"54{numero_serie}23F80D{timestamp_hex}01{temperature_hex}{temperature_hex}"

    # Create the outMsg dictionary in the specified format
    outMsg = {
        "payload": {
            "type": "uplink",
            "count": 0,
            "payload_cleartext": payload_cleartext,
            "device_properties": {
                "deveui": deveui  # Directly use deveui as a string
            },
            "protocol_data": {
                "port": 0,
                "rssi": 0,
                "sf": 0,
                "snr": 0,
            }
        }
    }
    return outMsg


# URL and headers for the POST request
url = 'http://vps.vaperail.com:1880/prevTemperature'  # Replace with the actual endpoint
headers = {'Content-Type': 'application/json'}
import requests
from requests.exceptions import ReadTimeout, RequestException

# Iterate over predictions
for sensor, predictions in predictions_dict.items():
    deveui = sensor_locations[sensor]['deveui']
    numero_serie = sensor_locations[sensor]['serial_number']
    print(f"Sensor: {sensor}, DevEUI: {deveui}, Serial Number: {numero_serie}")

    for prediction in predictions.itertuples():
        timestamp = pd.to_datetime(prediction.datetime).timestamp()
        temperature = prediction.rail_temp_pred

        print(f"Original Date and Time: {prediction.datetime}, Timestamp: {timestamp}, Temperature: {temperature}")

        outMsg = creer_outmsg(deveui, numero_serie, timestamp, temperature)

        print("outMsg =", json.dumps(outMsg, indent=4))

        # Send the POST request
        response = requests.post(url, json=outMsg, headers=headers, timeout=3)

        if response.status_code == 200:
            print(f"Prediction successfully sent to the endpoint for sensor {sensor} at {prediction.datetime}.")
        else:
            print(
                f"Failed to send the prediction for sensor {sensor}. Status code: {response.status_code}, Response: {response.text}")
            # raise an error to stop the execution
            raise Exception(f"HTTP Error: Status code {response.status_code}, Response: {response.text}")
