import joblib
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class LSTMWeatherModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_out, num_layers):
        super(LSTMWeatherModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_layer_size = n_hidden

        # Stacked LSTM layer
        self.lstm = nn.LSTM(n_input, n_hidden, num_layers, batch_first=True)

        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_out)


    def forward(self, x):
        # Stacked LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]

        # Regular feed forward network
        x = nn.SiLU()(self.linear1(x))
        x = nn.SiLU()(self.linear2(x))
        x = self.linear3(x)
        return x


def encode_cycle(cycle_index, cycle_length, to_numpy=False):
    cos_encoding = np.cos(cycle_index * 2 * np.pi / cycle_length)
    sin_encoding = np.sin(cycle_index * 2 * np.pi / cycle_length)
    if not to_numpy:
        return cos_encoding, sin_encoding
    return cos_encoding.to_numpy(), sin_encoding.to_numpy()


def model_inference(model, day_data, hours_to_predict):
    if day_data.shape[0] != 24:
        print('Invalid input data')
        raise Exception
    day = day_data['date'].dt.dayofyear[23]
    hour = day_data['date'].dt.hour[23]
    day_data = day_data.drop(columns='date').to_numpy()
    predictions = np.zeros((hours_to_predict, 8))
    day_data = torch.tensor([day_data], dtype=torch.float32)
    for i in range(hours_to_predict):
        prediction = model(day_data)
        predictions[i] = prediction.detach().numpy()

        # temporary solution
        day_cos, day_sin = encode_cycle(day, 365)
        hour_cos, hour_sin = encode_cycle(hour, 24)
        time_encoding = torch.tensor([day_cos, day_sin, hour_cos, hour_sin], dtype=torch.float32)
        # Add the time encoding to the input tensor and remoe the first element in the list!
        new_data = torch.cat((prediction, time_encoding[None, :]), axis=1)
        day_data = torch.cat((day_data[:, 1:], new_data[None, :]), axis=1)

        # increase time and/or day
        hour += 1
        if hour % 24 == 0 and hour != 0:
            day += 1

    return predictions


if __name__ == '__main__':
    # Testing
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
        "start_date": "2023-12-25",
        "end_date": "2023-12-25",
        "hourly": ["temperature_2m", "apparent_temperature", "rain", "snowfall", "surface_pressure",
                   "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "timezone": "Europe/Berlin"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

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

    df = pd.DataFrame(data=hourly_data)

    df['day_cos'], df['day_sin'] = encode_cycle(df['date'].dt.dayofyear, 365)
    df['hour_cos'], df['hour_sin'] = encode_cycle(df['date'].dt.hour, 24)
    print(df)
    model = torch.load('trained_models/weather_model.pth')

    n_predictions = 168

    predictions = model_inference(model, df, n_predictions)
    scaler = joblib.load('trained_models/scaler.save')
    predictions = scaler.inverse_transform(predictions)
    plt.plot(range(n_predictions), predictions[:, 0])
    plt.show()
