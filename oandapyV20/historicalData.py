import json
import pandas as pd
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.types import DateTime
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from basic import accountID, access_token, api

def get_equity_df (since, util, interval, price):
    cnt = 'mid' if price == 'M' else 'ask' if price == 'A' else 'bid' if price == 'B' else 'mid'
    params = {
        'from': DateTime(since).value,
        'to': DateTime(until).value,
        'price': price,
        'granularity': interval
    }
    r = instruments.InstrumentsCandles(instrument='USD_JPY', params=params)
    api.request(r)
    raw_list = []

    for raw in r.response['candles'] :
        raw_list.append([raw['time'], raw[cnt]['o'], raw[cnt]['h'], raw[cnt]['l'], raw[cnt]['c'], raw['volume']])
    raw_df = pd.DataFrame(raw_list, columns=['Time', f'Open_{cnt}', f'Hight_{cnt}', f'Low_{cnt}', f'Close_{cnt}', 'Volume'])
    return raw_df

start = datetime.strptime('2005-01-01', '%Y-%m-%d')
end = datetime.strptime('2019-10-31', '%Y-%m-%d')
month = 12 * 2 #上限5000レコード
interval = 'H4'
restart = 0

since = start + relativedelta(months=(restart))
until = start + relativedelta(months=(month + restart))
df = pd.DataFrame()

while True:
    if until > datetime.now():
        until = datetime.now()
        if relativedelta(since, until).month == 0 : break

    print(f'since:{ since }, until:{ until }')

    raw_a = get_equity_df(since, until, interval, 'A')
    raw_b = get_equity_df(since, until, interval, 'B')
    raw = pd.merge(raw_a, raw_b)
    raw.index = pd.to_datetime(raw.index)

    df = pd.concat([df, raw])

    since = until
    until = until + relativedelta(months=month)
    if since >= end : break
