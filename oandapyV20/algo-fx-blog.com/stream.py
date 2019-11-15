# http://www.algo-fx-blog.com/fx-api-rate-streaming-python/
import pandas as pd
import matplotlib
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
print(matplotlib.matplotlib_fname())
# backend: TkAgg
# backend: Qt5Agg

params = {
    "count": 200,
    "granularity": "M5"
}

# APIから為替レートのストリーミングを取得
r = instruments.InstrumentsCandles(instrument="USD_JPY", params=params)
api.request(r)

# ストリーミングの最初の1件目のデータを確認
print(r.response["candles"][0])

# 為替レートの dict を DataFrame へ変換
rate = pd.DataFrame.from_dict({r.response["candles"][i]["time"]: r.response["candles"][i]["mid"]
                            for i in range(0, len(r.response["candles"]))
                            for j in r.response["candles"][i]["mid"].keys()},
                        orient="index",
                      )

# インデックスの日付を綺麗にする
rate.index = pd.to_datetime(rate.index)

# DataFrameの確認
print(rate.head())

# データフレームからローソク足チャートへ
def candlechart(data, width=0.8):
    fig, ax = plt.subplots()
    # ローソク足
    mpf.candlestick2_ohlc(ax, opens=data.o.values, closes=data.c.values,
                          lows=data.l.values, highs=data.h.values,
                          width=width, colorup='r', colordown='b')

    # x軸を時間にする
    xdate = data.index
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))

    def mydate(x, pos):
        try:
            return xdate[int(x)]
        except IndexError:
            return ''

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')


    fig.autofmt_xdate()
    fig.tight_layout()

    return fig, ax

# ローソク足チャートのプロッティング
candlechart(rate)
