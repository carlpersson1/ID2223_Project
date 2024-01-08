import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Whether or not to upload the final model i.e. for testing
upload_model = False


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
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def create_windows(data, window_size, step=1):
    X, y = [], []
    for i in range(0, len(data) - window_size, step):
        # Create sequences (windows) of data
        X.append(data[i:(i + window_size)])
        # Append the target variable, the next value after the window
        y.append(data[i + window_size, :8])
    return np.array(X), np.array(y)


# You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
project = hopsworks.login()
fs = project.get_feature_store()
weather_fg = fs.get_feature_group(name="weather", version=1)

# Load the data
X_data = weather_fg.read()
raw_X_data = X_data.sort_values(by='date').drop(columns=['date'])
raw_X_data = raw_X_data.to_numpy()

# Scale the input data!
scaler = StandardScaler()
scaler.fit(raw_X_data[:, :8])
raw_X_data[:, :8] = scaler.transform(raw_X_data[:, :8])


# Create sequences of input data, each window consisting of one day of observations
X_data, Y_data = create_windows(raw_X_data, 24)

# Shuffle int0 train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data.copy(), Y_data.copy(), test_size=0.2, random_state=42,
                                                    shuffle=True)
# Model Hyperparameters
n_input = 12 # Number of features
n_hidden = 128  # Number of hidden nodes
n_out = 8 # Number of classes
num_layers = 2

# Training hyperparameters
epochs = 50
batch_size = 512
learning_rate = 0.0005
w_decay = 0.0005

# Load model to gpu
model = LSTMWeatherModel(n_input, n_hidden, n_out, num_layers)
device = torch.device("cuda")
model.to(device)

# Define loss function, optimizer, and convert data to tensors
loss_fun = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)
x_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
x_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(Y_test, dtype=torch.float32, device=device)

# Create TensorDatasets
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# Define dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_loss_array = []
val_loss_array = []

# Training loop
epoch_loop = tqdm.tqdm(range(epochs), desc=f"Epoch {1}/{epochs}")
for epoch in epoch_loop:
    # Training loss and grad updates
    train_loss = 0
    val_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = loss_fun(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation loss
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            val_loss += loss_fun(outputs, batch_y).item()
        val_loss /= len(test_loader)
        epoch_loop.set_description(f"Epoch {epoch + 1}/{epochs} [Train Loss: {train_loss:.4f}] [Validation Loss: {val_loss:.4f}]")

        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)

if upload_model:
    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()

    # The contents of the 'iris_model' directory will be saved to the model registry. Create the dir, first.
    model_dir = "trained_models"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_dir + '/weather_model.pth')

    # Save the standardizer since it is needed for inference!
    joblib.dump(scaler, model_dir + '/scaler.save')

    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(Y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    weather_model = mr.python.create_model(
        name="weather_model",
        metrics={"Validation MSE": val_loss_array[-1]},
        model_schema=model_schema,
        description="Weather predictor for Stockholm"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    weather_model.save(model_dir)
else:
    plt.plot(range(len(train_loss_array)), train_loss_array, label='Train loss')
    plt.plot(range(len(val_loss_array)), val_loss_array, label='Validation loss')
    plt.legend()
    plt.show()

    # Try to plot features over 2 days amd compare to true values
    test_data = raw_X_data.copy()[2000:2048]

    test_x_data, test_y_data = create_windows(test_data, 24)

    predictions = model(torch.tensor(test_x_data, dtype=torch.float32, device=device))

    predictions = scaler.inverse_transform(predictions.detach().cpu().numpy())
    outcomes = scaler.inverse_transform(test_y_data)
    plt.plot(range(24), predictions[:, 0], label='Predictions')
    plt.plot(range(24), outcomes[:, 0], label='Outcomes')
    plt.legend()
    plt.show()
