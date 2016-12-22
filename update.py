import urllib2
import StringIO
import gzip
import io
import csv
import sqlite3
import pandas as pd
import numpy as np


# Download updated CSV
resp   = urllib2.urlopen('http://api.bitcoincharts.com/v1/csv/krakenUSD.csv.gz')
comp   = StringIO.StringIO(resp.read())
comp.seek(0)

dc = gzip.GzipFile(fileobj=comp, mode='rb')
df = np.asarray(list(csv.reader(dc)))

conn = sqlite3.connect('bitcoin.db')
c = conn.cursor()

c.execute('drop table if exists k_usd')
c.execute('CREATE TABLE k_usd (date real, price real, quantity real)')

for row in df:
  c.execute('INSERT INTO k_usd VALUES (?,?,?)', (row[0],row[1],row[2]))


conn.commit()
conn.close()