import sys
sys.path.append('./lib')  

import pandas as pd
import datetime
from random import randint, random

from calc import calcProfit, getChunks

# Settings
file_name      = 'k_usd.csv' #http://api.bitcoincharts.com/v1/csv/krakenUSD.csv.gz
mode           = 'accuracy'

fund     = 1000
fee      = 0.26
accuracy = 0.55
count    = 60
trade_delay    = 3600 #3600 = 1 hour, 600 = 10 minutes
day_in_seconds = 86400

# Load CSV chunks
csv    = pd.read_csv("data/" + file_name, parse_dates=True)
chunks = getChunks(csv, count, day_in_seconds)

# Get average 
profit_total = 0
for idx, chunk in enumerate(chunks):
  # Calculate this chunks profit
  profit = calcProfit(fund, chunk, mode, trade_delay, fee, accuracy)

  # Add profit to fund
  fund += profit

  # Add profit to total
  profit_total += profit

  # Print this days profits
  print "Profit day {0}".format(idx)
  print '${:,.2f}'.format(profit)

profit_average = profit_total / count

print "Profit total over last {0} days".format(count)
print '${:,.2f}'.format(profit_total)

print "Profit average over last {0} days".format(count)
print '${:,.2f}'.format(profit_average)
