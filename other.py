import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

csv_path = "newkirk.csv"
df = pd.read_csv(csv_path, index_col="DATE")

# Check the data types of the columns
print(df.dtypes)
print(df["TMAX"].head())

# Convert "TMAX" column to numeric if needed
df["TMAX"] = pd.to_numeric(df["TMAX"])

#take values from x, predict y sliding window sort of
def df_to_x_y(df, window_size):
    df_as_np = df.to_numpy()
    x = [] #list
    y = [] #predicted value
    #ex [[[1],[2],[3],[4],[5]]] [6] -> 6 is predicted value from 1-5 
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        x.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(x), np.array(y) #return tuple

window_size = 5
x, y = df_to_x_y(df["TMAX"], window_size)
#print(x.shape, y.shape)

# Split the data into training and testing sets

x_train, y_train = x[:5000], y[:5000]
x_test, y_test = x[5000:], y[5000:]
x_val, y_val = x[5000:5200], y[5000:5200]

#print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

model1 = Sequential()
model1.add(InputLayer((window_size, 1))) #define input size
model1.add(LSTM(64))
model1.add(Dense(8, activation="relu"))
model1.add(Dense(1, activation='linear'))

model1.summary() #print model summary

cp = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(reduction='auto'), optimizer=Adam(learning_rate=0.00001), metrics=[MeanSquaredError()])
#higher the numer the faster the model will decrease loss
#we dont want this because we want to find local min

model1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, callbacks=[cp])

model1 = load_model('model1/')
train_predictions = model1.predict(x_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})

plt.plot(train_results['Train Predictions'], label='Train Predictions')
