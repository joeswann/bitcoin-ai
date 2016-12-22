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

print first_time

for h in range(days * 24):
  start_time = first_time + (h * hour)
  end_time   = start_time + hour*10

  rows = c.execute('SELECT * from k_usd where (date > ?) order by date desc', start_time).fetchall()

  print rows
  break

# print c.execute('SELECT * FROM k_usd order by date desc').fetchall()

conn.commit()
conn.close()