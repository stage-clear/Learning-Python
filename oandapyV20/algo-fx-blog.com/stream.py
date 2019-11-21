# http://www.algo-fx-blog.com/fx-api-rate-streaming-python/
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import mpl_finance as mpf
from matplotlib import ticker
import matplotlib.dates as mdates
import datetime
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import os, sys
sys.path.append(os.getcwd())
from basic import accountID, access_token, api

# https://qiita.com/typecprint/items/0d4bea1251ab3f816303
# print(matplotlib.matplotlib_fname())
# backend: TkAgg
# backend: Qt5Agg

params = {
    "count": 200,
    "granularity": "M5"
}

# APIから為替レートのストリーミングを取得
r = instruments.InstrumentsCandles(instrument="GBP_JPY", params=params)
api.request(r)

# ストリーミングの最初の1件目のデータを確認
print(r.response["candles"][0])

# 為替レートの dict を DataFrame へ変換
rate = pd.DataFrame.from_dict([ row['mid'] for row in r.response['candles'] ])
rate = rate.astype({'c': 'float64', 'l': 'float64', 'h': 'float64', 'o': 'float64'})
rate.columns = ['close', 'high', 'low', 'open']
rate['time'] = [ row['time'] for row in r.response['candles'] ]
rate['time'] = pd.to_datetime(rate['time']).astype(str)

# インデックスの日付を綺麗にする
rate.index = pd.to_datetime(rate['time'])

# DataFrameの確認
print(rate.head())

# データフレームからローソク足チャートへ
def candlechart(data, width=0.6):
    fig, ax = plt.subplots()

    print(data)
    # ローソク足
    mpf.candlestick2_ohlc(
        ax,
        opens=data.open.values,
        closes=data.close.values,
        lows=data.low.values,
        highs=data.high.values,
        width=width,
        colorup='#77d879',
        colordown='#db3f3f')

    xticks_number  = 12
    xticks_index   = range(0, len(data), xticks_number)
    xticks_display = [data.time.values[i][11:16] for i in xticks_index]
    # 時間を切り出すため、先頭12文字目から取る

    fig.autofmt_xdate()
    fig.tight_layout()

    plt.sca(ax)
    plt.xticks(xticks_index, xticks_display)
    plt.legend()
    plt.show()
    #return fig, ax

# ローソク足チャートのプロッティング
candlechart(rate)
