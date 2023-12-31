# ID2223 Final Project

This project deals with the task of predicting the weather in Stockholm. It utilizes the OpenMeteo API as the data source and fetches weather data to train and do inference. We use PyTorch to implement the model and Hopsworks for the feature store. For the UI, a HuggingFace Spaces app is created that gives the current weather forecast for the next day.

# Feature pipeline

The feature pipeline fetches the weather data for Stockholm from OpenMeteo, by giving the coordinates and the time span of past weather data. In this case, we selected the weather data from 1. January 2018 to the 25th December 2023 (the data the training data was fetched) (5 year span). The response provides hourly weather data for the following weather variables:  temperature for 2 meters above ground, apparent temperature, precipitation, rain, snowfall, surface pressure, cloud cover, cloud cover low, cloud cover mid, cloud cover high, wind speed 10 meters above ground, wind speed 100 meters above ground, wind direction 10m, wind direction 100m, wind gusts 10m.

Next a dataframe is created that contains the weather data together with the date it occurred (including hour, such that it is indexable by hour). We perform data validation (removing NaNs) and EDA to do feature engineering. Since the data is already in a pretty good state, no extensive measures are necessary to prepare the data. We have plotted multiple variables to see if they make sense and how they develop over the years, starting with the temperature. The temperature follows a repeating wave pattern and temperatures are higher in the summer and lower in the winter. Overall it stays the same over the years, though it can be noted that 2023 was not as warm as the other years. Also, in the summer the average temperature is 18-20°C and in the winter it is 0-(-1)°C. For the rain, we observed that 2021 and 2023 were highest, whereas 2020 had a single day that had over 10mm, which is the maximum from the whole dataset. For snow, it can be observed that snow is present during every year around the winter. 

With Additional EDA, using the correlation matrix of the weather variables we found that windspeeds/windgusts at different heights correlate highly, and thus aren't as interesting to use as features and are therefore removed. Furthermore, the different cloud coverage - low, mid and high are not very interesting either. We removed these and kept the cloud coverage variable. In addition, precipition is removed as it is a combination of rain and snowfall and therefore unnecessary to be used as a feature.

In many datasets, dates and times are crucial features, but their standard numerical representations can be misleading for machine learning models. This is because time variables like days and hours are cyclical. For example, in a yearly cycle, day 1 (January 1st) is very close to day 365 (December 31st), but their numeric values are far apart. A similar concept applies to hours in a day.

To accurately represent this cyclical nature, we use sine and cosine functions to encode the days of the year and the hours of the day. This transformation maps these time features onto a circle, ensuring that values close in time are also close in their transformed space. This method enhances the model's ability to recognize and utilize these temporal relationships during training.

The data gets uploaded to the feature store stored as a feature group in Hopsworks, which is going to be used for the training.

# Training pipeline

In the training pipeline we load the features, process them and train a model, which is then pushed to the model registry. The features are standardized by removing the mean and scaling to unit variance with `StandardScaler()`. 

The first model we have tried to use is a simple neural network, which performed poorly. Therefore, we switched to an LSTM model, which 
performed considerably better. The stacked LSTM is fairly similar to the linear layer, but
is capable of holding short term memory as well. This is done by inputting a sequence of data
into the model, which allows it to make inference based on data further back in time. Making it
a simple but powerful layer for weather prediction purposes.

The stacked LSTM function requires an input sequence in order to make use of the short 
term memory part of the layer. This was done by creating 'windows' of data
where each input corresponds to a sequence of 24 hours of input data. Where the values to predict
are the features for the next hour i.e. the 25th hour in the sequence, which is not input into the
model.

Since the model predicts all the input features, but one hour in advance, we can use the output
of the model in combination with the previous 23 hours to predict two hours in advance, and so on.
This allows the model to predict arbitrarily far into the future, although with compounding errors.

The model is trained for 50 epochs on 80% of the data, by using AdamW with a batch size of 512. The learning rate is 0.0005 and weight decay is used as well with 0.0005. As the loss function we used the mean squared error. We achieve relatively good performance in 50 epochs, with a training error of close to 0.1 and a validation loss of around 0.18.


# Daily inference pipeline

In the daily inference pipeline, the model is used to make predictions for the next 24 hours. The results are then used for the app to display the current weather forecasts. The general process is to get the model from the model registry and the weather data from the feature group stored in the feature store. The model predicts the weather variables for the next 24 hours and the scaler, which was saved in the training pipeline is used to revert the values back to their original form (from normalized to original values). 

Since we already know that the model is supposed to predict the weather one hour in advance, we 
do not need to have it infer the timestamp of the generated features. Instead this is done artifically
by encoding the time/date + 1 hour, into the sine and cosine representation for use in predicting the next 
hour after that.

In addition the performance of the last prediction, so from the day before, is evaluated, by comparing it to the weather data that was recorded by OpenMeteo from that day. Since, the feature pipeline has fetched the data already for the previous day, it can be used to compare and evaluate the performance of our model. For both the current prediction (next 24 hours) and the last prediction evaluation (previous 24 hours) figures are created to be stored in the feature store and used for visual representation in the app. 

# Daily feature pipeline

In the data feature pipeline the goal is to continuously fetch data from OpenMeteo. For that purpose, we look at the latest weather data that is stored in the feature group and update it by fetching all the newly available weather data up to the time (or date) the program is run. This process allows us to get more data for training to improve the model, as well as test its capabilities. Before the new data is inserted into the feature group, it is first checked if any overlaps exist with the data currently stored in the feature group. This is done by checking if any dates overlap, which are removed. After that the data is inserted to the feature group. This program is run daily.

# HugginFace Spaces App

Our [app](https://huggingface.co/spaces/carlpersson/WeatherPrediction) showcases the predictions for the next 24 hours by using the daily inference pipeline. The  temperature in °C, apparent temperature in °C, rain in mm, snowfall in cm, surface pressure in hPa, cloud cover in %, wind speed in km/h and wind direction in degrees are predicted and updated everyday. These predictions are based on the latest data from the dataset, which is updated every day with two day old data. The predictions are based on the weather data from Stockholm specifically and the model predictions are unfortunately already outdated by the time they are made, due to the delay in data updates of our database. In addition to the predictions for the next 24 hours, the previous predictions (last 24 bours) together with the actual outcome of the weather (fetched from OpenMeteo) is merged, so that the accuracy of the predictions of the past day can be observed. 

# Running the code

In order to run the code, firstly run the feature_pipeline.py. This code (given that it
has access to a Hopswork account API key) will upload the cleaned weather data between 2018 and now 
to a feature group in Hopsworks. By running the training_pipeline.py file, the weather data 
will be downloaded and used to train a LSTM weather prediction model which is then uploaded to hopsworks.

Next the files deploy daily_feature_pipeline.py and daily_inference_pipeline.py need to be deployed
on modal, or any similar service (might require small changes in code). These files
will download the latest weather data daily and do batch inference to predict the next 
24 hours in the sequence. Images showing the prediction and previous performance is uploaded
to Hopsworks, which can then be viewed by uploading the app.py and requirements.txt to a
Huggingface space and deploying it.
