import pandas as pd
from random import randint, random

def getChunks(csv, counter, seconds):

  # First and last row in csv
  first_time = csv[:1].iloc[0]['time']
  last_time  = csv[-1:].iloc[0]['time']

  # Slice ranges
  start_time = last_time - seconds
  end_time   = last_time
  chunks = []
  while(counter > 0 and start_time > first_time):
    # Push day slice to days
    chunks.append(csv[(csv['time'] > start_time) & (csv['time'] < end_time)])

    # Move slice
    start_time -= seconds
    end_time   -= seconds

    # Count down to 0
    counter -= 1

  return chunks

def calcProfit(fund, chunk, mode, delay, fee=0.25, accuracy=1):
  index = 0
  pref = False
  current_fund = fund
  #either entire balance is usd or btc 
  balance = 'usd' # Assume usd at start of chunk

  # If buy and hold
  if(mode == 'hold'):
    first_price = chunk[:1].iloc[0]['price']
    last_price  = chunk[-1:].iloc[0]['price']
    diff = last_price / first_price
    return (diff * fund) - fund

  # Everything else
  for row,cur in chunk.iterrows():
    # Read next row
    if(index != 0):
      cur_price  = float(prev[1])
      next_price = float(cur[1])

      cur_time  = float(prev[0])
      next_time = float(cur[0])

      # Wait 10 minutes between trades
      if(next_time - cur_time < delay):
        continue

      choice = makeChoice(cur_price, next_price, mode, fee, accuracy)

      diff = 1 # Do nothing if choice == 0
      if(choice == 1 and balance == 'usd'): # Buys bitcoin if balance is in usd
        diff = (next_price / cur_price) * (1 - fee/100)
        balance = 'btc'
      elif(choice == 2 and balance == 'btc'): # Sells bitcoin if balance is in btc
        diff = (cur_price / next_price) * (1 - fee/100)
        balance = 'usd'

      # Multiply our fund by the diff
      current_fund *= diff

    prev = cur
    index += 1
  return current_fund - fund

def makeChoice(cur_price, next_price, mode, fee, accuracy):
  choice = 0 # Hold

  # Make correct choice because we know the future
  if(cur_price < next_price): # BTC will go up, so buy
    diff = (next_price / cur_price) * (1 - fee/100) #Apply fee
    if(diff > 1): choice = 1
  if(cur_price > next_price): # BTC will go up, so sell
    diff = (cur_price / next_price) * (1 - fee/100) #Apply fee
    if(diff > 1): choice = 2

  # Make random choice
  if(mode == 'random'):
    choice = randint(0,2)

  # % Chance of making correct choice
  if(mode == 'accuracy'):
    new_choice = randint(0,2)

    # If we should make a wrong choice and new choice is wrong
    if(random() > accuracy and new_choice == choice):
      # Make sure we pick something else
      while(new_choice == choice): 
        new_choice = randint(0,2)
      # Update choice
      choice = new_choice

  # if(mode == 'predict'):
    # This is where our TF model comes in

  return choice;