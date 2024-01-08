import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from utils import encode_cycle
import hopsworks

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
# Get data over a year
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 59.3294,
	"longitude": 18.0687,
	"start_date": "2018-01-01",
	"end_date": pd.Timestamp.now().strftime('%Y-%m-%d'),
	"hourly": ["temperature_2m", "apparent_temperature", "rain", "snowfall", "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
	"timezone": "Europe/Berlin"
}
responses = openmeteo.weather_api(url, params=params)

response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(7).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s"),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

# Fill nans with last valid observation
hourly_dataframe = pd.DataFrame(data = hourly_data).fillna(method='ffill')

# Encode day of year and time of day into cyclical representations
hourly_dataframe['day_cos'], hourly_dataframe['day_sin'] = encode_cycle(hourly_dataframe['date'].dt.dayofyear, 365)
hourly_dataframe['hour_cos'], hourly_dataframe['hour_sin'] = encode_cycle(hourly_dataframe['date'].dt.hour, 24)

# Upload data to hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

weather_fg = fs.get_or_create_feature_group(
    name="weather",
    version=1,
    primary_key=['date', 'temperature_2m', 'apparent_temperature', 'rain', 'snowfall', 'surface_pressure',
                 'cloud_cover', 'wind_speed_10m', 'wind_direction_10m', 'day_cos', 'day_sin', 'hour_cos', 'hour_sin'],
    description="Weather dataset")

weather_fg.insert(hourly_dataframe)