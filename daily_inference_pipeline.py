import os
import modal
import hopsworks
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("weather_inference_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "numpy", "joblib", "torch", "openmeteo_requests",
                                                  "requests_cache", "retry_requests", "matplotlib", "scikit-learn"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def encode_cycle(cycle_index, cycle_length, to_numpy=False):
    cos_encoding = np.cos(cycle_index * 2 * np.pi / cycle_length)
    sin_encoding = np.sin(cycle_index * 2 * np.pi / cycle_length)
    if not to_numpy:
        return cos_encoding, sin_encoding
    return cos_encoding.to_numpy(), sin_encoding.to_numpy()


def model_inference(model, scaler, day_data, hours_to_predict):
    if day_data.shape[0] != 24:
        print('Invalid input data')
        raise Exception
    day = day_data['date'].dt.dayofyear.tail().to_numpy()[0]
    hour = day_data['date'].dt.hour.tail().to_numpy()[0]
    day_data = day_data.drop(columns='date').to_numpy()
    day_data[:, :8] = scaler.transform(day_data[:, :8])
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


def g():
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    dataset_api = project.get_dataset_api()

    # Load the dataset and find the latest weather feature
    weather_fg = fs.get_feature_group(name="weather", version=1)
    df = weather_fg.read()
    df = df.sort_values(by='date')
    model = mr.get_model("weather_model", version=1)
    model_dir = model.download()

    model = torch.jit.load(model_dir + '/weather_model.pth')
    scaler = joblib.load(model_dir + '/scaler.save')

    # Do the prediction of the next 24 hours
    prediction = model_inference(model, scaler, df.iloc[-24:], 24)
    prediction = scaler.inverse_transform(prediction)

    # Evaluate the performance of the last prediction
    prediction_eval = model_inference(model, scaler, df.iloc[-48:-24], 24)
    prediction_eval = scaler.inverse_transform(prediction_eval)

    last_datapoint = df['date'].max().to_pydatetime()
    x1 = [last_datapoint + datetime.timedelta(hours=i) for i in range(24)]
    x2 = [last_datapoint + datetime.timedelta(hours=i) - datetime.timedelta(hours=24) for i in range(24)]

    # Create figures for app
    fig_dir = 'app_figures'
    if os.path.isdir(fig_dir) == False:
        os.mkdir(fig_dir)

    for i, col in enumerate(df.columns[1:9]):
        plt.plot(x1, prediction[:, i])
        plt.gcf().autofmt_xdate()
        plt.savefig(fig_dir + '/pred_' + col + '.png')
        plt.clf()

        # Upload the images to the dataset api in hopsworks
        dataset_api.upload(fig_dir + '/pred_' + col + '.png', 'Resources/predictions', overwrite=True)

        plt.plot(x2, prediction_eval[:, i], label='Prediction')
        plt.plot(x2, df.iloc[-24:][col], label='Outcome')
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.savefig(fig_dir + '/prev_' + col + '.png')
        plt.clf()

        # Upload the images to the dataset api in hopsworks
        dataset_api.upload(fig_dir + '/prev_' + col + '.png', 'Resources/predictions', overwrite=True)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("weather_inference_daily")
        with stub.run():
            f()