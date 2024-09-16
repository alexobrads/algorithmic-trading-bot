# Library imports
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pandas_ta as ta
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import models as al


# Import the data
df = al.data.get_historical_bars()

# Calculating some technical indicators
# Simple Moving Averages
sma10 = df.ta.sma(length=10, append=True)
sma50 = df.ta.sma(length=50, append=True)
sma100 = df.ta.sma(length=100, append=True)
# Momentum
mom = df.ta.mom(append=True)
# MACD
macd = df.ta.macd(append=True)
# Bollinger Bands
bbands20 = df.ta.bbands(length=20, append=True)
# Stochastic RSI
stochrsi = df.ta.stochrsi(append=True)

# WRITE SOME CODE TO GRAB ALL INDICATORS THAT PANDAS_TA CAN PROVIDE
# CAN DO VARIABLE SELECTION LATER OR JUST LET NN HANDLE IT
# test = df.ta.strategy(ta.AllStrategy)


# Manipulating the dataset

# Creating targets - close price in 2hrs
close_2hr_ahead = df.close.shift(-2)
diff_close_2hr_ahead = close_2hr_ahead - df["close"]

# Checking with a copy of df
df_copy = df.copy(deep=True)
df_copy.insert(4, "diff_close_2hr_ahead", diff_close_2hr_ahead)
#df_copy
# Looks correct

# Grabbing the index to use later
df_idx = df["end"]

# Dropping index (differencing will make it useless)
df.drop("end", axis=1, inplace=True)

# Differencing to remove trend
df_diff = df.diff()

# Appending targets
df_diff.insert(4, "diff_close_2hr_ahead", diff_close_2hr_ahead)

# Removing NAs
df_diff = df_diff.iloc[100:-2, :]

# Resetting index
df_diff.reset_index(inplace=True, drop=True)
# Plotting histograms
df_diff.hist(figsize=(15,10), bins=100)


# Standardizing

# Getting train set size
train_size = math.floor(len(df_diff)*0.8)

# Getting train set mean and std
def standardize(dataframe, train_size):
    x_bars, x_stds = [], [] # storage so we can un-transform later
    for i in range(len(dataframe.columns)):
        col = dataframe.iloc[:train_size,i] # Look at only train_size datapoints to calculate mean & std
        x_bar, x_std = col.mean(), col.std()
        x_bars.append(x_bar)
        x_stds.append(x_std)
        dataframe.iloc[:,i] = dataframe.iloc[:,i].map(lambda x_t: (x_t - x_bar) / x_std)
    x_bars = pd.DataFrame([x_bars], columns=dataframe.columns)
    x_stds = pd.DataFrame([x_stds], columns=dataframe.columns)
    return dataframe, x_bars, x_stds

df_std, x_bars, x_stds = standardize(df_diff, train_size)

# Plotting histograms to check standardization worked
df_std.hist(bins=100, figsize=(20,15))


# Find percentiles

percentiles = [25, 50, 75]
percentile_values = []
np_diff_close_2hr_ahead = np.array(df_std.diff_close_2hr_ahead)
for i in percentiles:
    percentile_value = np.percentile(np_diff_close_2hr_ahead, i)
    percentile_values.append(percentile_value)

print(percentile_values)

ax = df_std.diff_close_2hr_ahead.hist(bins=100)
for i in percentile_values:
    plt.axvline(x=i, color = "red", linestyle="dashed", alpha=0.5)


# Assigning classes to the data

target_classes = []
for i in range(len(df_std)):
    # 0th to 25th percentiles   - Class 0
    if df_std.diff_close_2hr_ahead[i] <= percentile_values[0]:
        target_classes.append(0)
    # 25th to 50th percentiles  - Class 1
    elif percentile_values[0] < df_std.diff_close_2hr_ahead[i] <= percentile_values[1]:
        target_classes.append(1)
    # 50th to 75th percentiles  - Class 2
    elif percentile_values[1] < df_std.diff_close_2hr_ahead[i] <= percentile_values[2]:
        target_classes.append(2)
    # 75th to 100th percentiles - Class 3
    else:
        target_classes.append(3)

# Adding the column to the dataframe
df_std['targets'] = target_classes

# Converting to categorical datatype
#df_std['targets'] = df_std['targets'].astype('category')

# Dropping the diff_close_2hr_ahead column - This is effectively the output too
df_std.drop("diff_close_2hr_ahead", axis=1, inplace=True)

# Checking class assignment
df_std.targets.hist()


# Preparing data for the NN

# Train-test split
train_size = math.floor(len(df_diff)*0.8)
test_size = len(df) - train_size
df_train = df_std.iloc[:train_size,:]
df_test = df_std.iloc[train_size:, :]

# Dataloaders
# Defining inputs and targets
df_inputs_train = df_train.drop(["targets"], axis=1)
df_targets_train = df_train["targets"]
df_inputs_test = df_test.drop(["targets"], axis=1)
df_targets_test = df_test["targets"]
# Converting from dataframe
inputs_train = torch.from_numpy(df_inputs_train.values)
targets_train = torch.from_numpy(df_targets_train.values)
inputs_test = torch.from_numpy(df_inputs_test.values)
targets_test = torch.from_numpy(df_targets_test.values)
# Converting to a dataset object
ds_train = TensorDataset(inputs_train.float(), targets_train)
ds_test = TensorDataset(inputs_test.float(), targets_test)

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=test_size, shuffle=False)

# Checking dataloader
for i, (inputs, targets) in enumerate(dl_train):
    print("input shape:", inputs.shape)
    #print("inputs:", inputs)
    print("target shape:", targets.shape)
    #print("targets:", targets)
    break


# MODEL CLASS

# Linear Model - Classification
class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0):
        super(NeuralNet, self).__init__()
        # Defining parameters
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # Here we define the layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax() # Is the softmax actually necessary here? Sperm project didn't have it.

    def forward(self, x):
        # Here we define how the data will pass through the layers
        x = self.fc1(x)
        x = self.bn1(x) # BN before activation according to https://forums.fast.ai/t/why-perform-batch-norm-before-relu-and-not-after/81293/4
        x = self.relu(x)
        x = self.dropout1(x) # Dropout after ReLU according to https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        #x = self.softmax(x)
        return x


# TRAINING LOOP

# Setting hyperparameters
num_epochs = 500
learn_rate = 1e-6

# Defining the model from the class we defined
model = NeuralNet(input_dim=18, num_classes=4, dropout=0.2)
# Setting device to GPU and sending the model
device = torch.device('cuda:0')
model.to(device) # Send the model to GPU

# Optimizer and loss
optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dl_train):
        model.train() # The model must be trained in training mode
        # Send data to GPU
        if i==0 and epoch==0:
            print("input shape: ", inputs.shape)
            print("target shape: ", targets.shape)
        inputs = inputs.to(device)
        targets = targets.to(device)


        # Forward pass
        outputs = model(inputs) # Generate predictions from model
        loss = criterion(outputs, targets) # Calculate loss

        # Backward and optimize
        optimizer.zero_grad() # Clear gradients from last time we did this
        loss.backward() # Backpropogate the gradients
        optimizer.step() # Update weights


    # Printing progress
    print('Epoch [{}/{}] - Loss: {}'.format(epoch+1, num_epochs, loss))


# Generating outputs
model.eval()
for i, (inputs, targets) in enumerate(dl_test):
    print("inputs shape:", inputs.shape)
    inputs = inputs.to(device)
    print("outputs shape:", targets.shape)
    test_preds = model(inputs)
    _, test_preds = torch.max(test_preds.data, 1)

test_preds = test_preds.detach().cpu()
test_preds = pd.DataFrame(test_preds.numpy(), columns=['preds'])

# Viewing distribution of predicted classes
test_preds.hist()


# Unstandardizing the test set predictions -  NOT REQUIRED FOR CLASSIFICATION MODEL

def unstandardize(dataframe, x_bars, x_stds, target_col=None):
    dataframe_unstd = dataframe.copy(deep=True) # So we don't overwrite the original dataframe
    # True data case where we have multiple columns
    if len(dataframe_unstd.columns) > 1:
        col_names = dataframe_unstd.columns
        for c in range(len(dataframe_unstd.columns)): # could also do 'for c in col_names:'
            dataframe_unstd[col_names[c]] = dataframe_unstd[col_names[c]] * x_stds[col_names[c]].values + x_bars[col_names[c]].values
        return dataframe_unstd
    # Predictions case where we only have one column of outputs
    else:
        dataframe = dataframe * x_stds[target_col].values + x_bars[target_col].values
        return dataframe

test_preds_unstd = unstandardize(test_preds, x_bars, x_stds, "diff_close_2hr_ahead")

# Checking that unstandardization worked
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 8))
ax1.hist(test_preds['preds'], bins=50)
ax2.hist(test_preds_unstd['preds'], bins=50)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 8))
ax1.hist(df_test.diff_close_2hr_ahead)
ax2.hist(test_preds_unstd['preds'])