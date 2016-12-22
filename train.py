import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys

# Settings
file_name      = 'k_usd.csv' #http://api.bitcoincharts.com/v1/csv/krakenUSD.csv.gz
day_in_seconds = 86400
hour_in_seconds = 3600
two_hours_in_seconds = hour_in_seconds*2
days           = 10

# Load CSV chunks
print "Reading CSV"
csv = pd.read_csv("data/" + file_name, parse_dates=True) 

# Grab only recent data, last 180 days or so
end_time   = csv[-1:].iloc[0]['time']
start_time = (end_time - (days * day_in_seconds))
dataset    = csv[(csv['time'] > start_time) & (csv['time'] < end_time)]

# Group into 120 minute chunks
chunk_max = ((days * day_in_seconds) / two_hours_in_seconds)

chunk_counter = 0
chunk_max_size = 0

print "Getting max chunk length"
while chunk_counter <= chunk_max:
  # first  60 minutes = input
  # Last number @ 120 = output
  s_time = start_time + (chunk_counter * hour_in_seconds * 2)
  m_time = s_time + hour_in_seconds

  first_chunk_df = csv[(csv['time'] > s_time) & (csv['time'] < m_time)]
  chunk_length = len(first_chunk_df)
  if(chunk_length > chunk_max_size): chunk_max_size = chunk_length
  chunk_counter += 1


chunk_counter = 0
print "Formatting chunks"
inputs = []
outputs = []
while chunk_counter <= chunk_max:
  # first  60 minutes = input
  # Last number @ 120 = output
  s_time = start_time + (chunk_counter * hour_in_seconds * 2)
  m_time = s_time + hour_in_seconds
  e_time = s_time + (hour_in_seconds*2)

  first_chunk_df = csv[(csv['time'] > s_time) & (csv['time'] < m_time)]
  last_chunk_df  = csv[(csv['time'] > m_time) & (csv['time'] < e_time)]

  # Get chunk length
  chunk_length = len(first_chunk_df)

  if(chunk_length > 10):
    # Instead of prices, we want the difference between each price and the next
    c_raw = first_chunk_df.reset_index()
    # df = first_chunk_df.reset_index()
    # print len(c_input)
    c_input = np.zeros(((len(c_raw) - 1), 1, 1))
    for idx, c_cur in c_raw.iterrows():
      if(idx + 1 < chunk_length): 
        c_next_price = c_raw["price"][idx+1] 
        c_input[idx,0,0] = c_cur["price"] / c_next_price

    cur_price = float(c_raw["price"][-1:])
    next_price = float(last_chunk_df["price"][-1:])

    # Output should be
    # 1  = buy
    # 0  = hold
    # -1 = sell

    fee = 0.26
    choice = 0    
    if(cur_price < next_price): # BTC will go up, so buy
      diff = (next_price / cur_price) * (1 - fee/100) #Apply fee
      if(diff > 1): choice = 1
    if(cur_price > next_price): # BTC will go up, so sell
      diff = (cur_price / next_price) * (1 - fee/100) #Apply fee
      if(diff > 1): choice = -1

    inputs.push(c_input)
    outputs.push(choice)

  chunk_counter += 1


# print chunks

# Select train/test data
# train, test = train_test_split(chunks, test_size = 0.25)


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 5
epochs = 500
# number of elements ahead that are used to make the prediction
lahead = 1

## TRAINING
# Input = first hour data
# Output = choice for end of second hour
print "Training"
chunk    = chunks[0]
c_input  = chunk["input"]
c_output = chunk['output']
print('Input shape:', c_input.shape)

expected_output = np.zeros((1, 1))
expected_output[0, 0] = c_output

print('Output shape')
print(expected_output.shape)

print('Creating Model')
model = Sequential()
# model.add(LSTM(50,
#                batch_input_shape=(batch_size, tsteps, 1),
#                return_sequences=True,
#                stateful=True))
# model.add(LSTM(50,
#                return_sequences=False,
#                stateful=True))
# model.add(Dense(output_dim=1))
model.add(Dense(input_shape=(),output_shape=expected_output.shape))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(c_input,
              expected_output,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
predicted_output = model.predict(c_input, batch_size=batch_size)

print('Plotting Results')
plt.subplot(2, 1, 1)
plt.plot(expected_output)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.savefig("keras1")