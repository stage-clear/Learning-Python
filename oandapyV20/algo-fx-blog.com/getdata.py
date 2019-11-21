import pandas as pd

import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import os, sys
sys.path.append(os.getcwd())
from basic import accountID, api

instrument = 'GBP_JPY'
params = {
    'alignmentTimezone': 'Japan',
    'count': 200,
    'granularity': 'M5'
}

r = instruments.InstrumentsCandles(instrument=instrument, params=params)
api.request(r)
candles_data = r.response['candles']

rate_df = pd.DataFrame.from_dict([row['mid'] for row in candles_data])
rate_df = rate_df.astype({'c': float, 'l': float, 'h': float, 'o': float})
rate_df.index = [row['time'] for row in candles_data]

print(rate_df.head())
print(rate_df.tail())
print(rate_df.info())
