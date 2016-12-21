# Bitcoin Experiments
Just messing around with bitcoin exchanges. Maybe something will stick?

##Update (update.py)
Downloads a fresh copy of the exchange CSV (currently Kraken USD).

##Benchmark (benchmark.py)
Run some testbench code to run a set of strategies against a bitcoin exchange CSV. At the moment it runs the last 50 days in 24 hour intervals with trades occuring every 10 minutes.
- Buy and hold
- Perfect (100% accuracy)
- Random
- Accuracy (random w/ bias)

##Todo
- Train an RNN against 1 hour blocks of data. 
  - Use previous 60 minutes of data as input and a [buy/sell/hold] classifier as output
- Gauge accuracy of predictor. 
  - If accuracy consistently beats plain buy and hold, integrate w/ gekko
- Retrain every 24h against latest data