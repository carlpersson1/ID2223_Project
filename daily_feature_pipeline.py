import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("weather_daily")

   image = modal.Image.debian_slim().pip_install(["hopsworks", "numpy"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()


def encode_cycle(cycle_index, cycle_length, to_numpy=False):
    import numpy as np
    cos_encoding = np.cos(cycle_index * 2 * np.pi / cycle_length)
    sin_encoding = np.sin(cycle_index * 2 * np.pi / cycle_length)
    if not to_numpy:
        return cos_encoding, sin_encoding
    return cos_encoding.to_numpy(), sin_encoding.to_numpy()


def get_new_data(latest_date):
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    from_date = latest_date.strftime('%Y-%m-%d')
    to_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    # Get data over a year
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 59.3294,
        "longitude": 18.0687,
        "start_date": from_date,
        "end_date": to_date,
        "hourly": ["temperature_2m", "apparent_temperature", "rain", "snowfall", "surface_pressure",
                   "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
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

    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # Drop potential data containing NANS
    hourly_dataframe = hourly_dataframe.dropna()

    # Encode day/hour cycle
    hourly_data["day_cos"], hourly_data["day_sin"] = encode_cycle(hourly_dataframe['date'].dt.dayofyear, 365)
    hourly_dataframe['hour_cos'], hourly_dataframe['hour_sin'] = encode_cycle(hourly_dataframe['date'].dt.hour, 24)

    return hourly_dataframe


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    # Load the dataset and find the latest weather feature
    weather_fg = fs.get_feature_group(name="weather", version=1)
    df = weather_fg.read()
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    latest_date = df['date'].max()

    # Get data from latest weather feature to now
    weather_df = get_new_data(latest_date)

    # Make sure that there are no duplicates in data:
    overlap = weather_df[weather_df['date'].isin(df['date'])]
    # Remove any overlap if there is any!
    weather_df = weather_df[~weather_df['date'].isin(overlap['date'])]

    # Insert the new data into the dataset
    weather_fg.insert(weather_df)


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("iris_daily")
        with stub.run():
            f()