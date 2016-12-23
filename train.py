import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys

# fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)

# Split data into 24 hour periods
minute = 60
hour = minute * 60
day  = hour * 24
days = 3
fee  = 0.26

conn       = sqlite3.connect('bitcoin.db')
c          = conn.cursor()
first_time = float(c.execute('SELECT date FROM k_usd order by date desc limit 1').fetchone()[0])
df_raw     = pd.read_sql_query('SELECT * from k_usd order by date desc', conn) 
conn.commit()
conn.close()

# Format dataset
h_range = days * 24
dataset = np.zeros((h_range,121))
for h in range(days * 24):
  s_time = first_time - (h * hour) - (hour * 2) # Beginning of segment 
  m_time = s_time + hour     # End of 1 hour segment
  e_time = s_time + hour * 2 # 2 Hour mark

  s_price = float(df_raw[(df_raw["date"] <= s_time)][:1].iloc[0]['price'])
  m_price = float(df_raw[(df_raw["date"] <= m_time)][:1].iloc[0]['price'])
  e_price = float(df_raw[(df_raw["date"] <= e_time)][:1].iloc[0]['price'])

  # Last row is the buy/sell decision
  choice = 0 # Hold
  # Make correct choice because we know the future
  if(m_price < e_price): # BTC will go up, so buy
    diff = (e_price / m_price) * (1 - fee/100) #Apply fee
    if(diff > 1): choice = 1

  if(m_price > e_price): # BTC will go up, so sell
    diff = (m_price / e_price) * (1 - fee/100) #Apply fee
    if(diff > 1): choice = -1

  # Save choice in last column
  dataset[h,120] = choice

  # We want 121 columns
  # Each column is 30 seconds
  for t in range(120):
    t_pointer    = s_time + (t*30)
    dataset[h,t] = df_raw[(df_raw["date"] <= t_pointer)]["price"][:1]

  print "Formatted hour {0} of {1}".format(h,h_range)

# Split dataset
X = dataset[:,0:120]
Y = dataset[:,120]    

# Create model
model = Sequential()
model.add(Dense(120, input_dim=120, init='uniform', activation='relu'))
model.add(Dense(80, init='uniform', activation='relu'))
model.add(Dense(80, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=300, batch_size=100)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))