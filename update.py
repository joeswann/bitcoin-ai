import urllib2
import StringIO
import gzip
import io

# Download updated CSV
resp   = urllib2.urlopen('http://api.bitcoincharts.com/v1/csv/krakenUSD.csv.gz')
comp   = StringIO.StringIO(resp.read())
comp.seek(0)

decomp = gzip.GzipFile(fileobj=comp, mode='rb')

with open('data/k_usd.csv', 'w') as outfile:
  outfile.write("time,price,quantity\n" +  decomp.read())

print decomp.read()