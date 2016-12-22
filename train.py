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

# Split data into 24 hour periods
minute = 60
hour = minute * 60
day  = hour * 24

days = 30

conn = sqlite3.connect('bitcoin.db')
c = conn.cursor()

first_time = float(c.execute('SELECT date FROM k_usd order by date desc limit 1').fetchone()[0])

for h in range(days * 24):
  s_time = first_time - (h * hour) - (hour * 2) # Beginning of segment 
  m_time = s_time + hour     # End of 1 hour segment
  e_time = s_time + hour * 2 # 2 Hour mark

  df = pd.read_sql_query('SELECT * from k_usd where date > {0} and date < {1} order by date desc'.format(s_time, e_time), conn)
  s_row = df[:1]
  e_row = df[-1:]

  # We want 121 columns
  # Each column is 30 seconds
  prices =  np.zeros((1,121,1))
  for t in range(120):
    t_pointer = s_time + (t*30)
    prices[0,t,0] = df[(df["date"] <= t_pointer)]["price"]
    
  print prices
# print c.execute('SELECT * FROM k_usd order by date desc').fetchall()

conn.commit()
conn.close()