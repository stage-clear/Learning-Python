# http://www.algo-fx-blog.com/fx-api-oanda-v20-histrocial-rate-pandas-csv/
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import os, sys
sys.path.append(os.getcwd())
from basic import accountID, access_token, api


# APIに渡すパラメータの設定
params = {
    "count": 2000,
    "granularity": "S5" # 5 seconds
}

# APIへ過去為替レートをリクエスト
r = instruments.InstrumentsCandles(instrument="USD_JPY", params=params)
api.request(r)

# APIから取得した最初のmid（中値）を確認
print(r.response["candles"][1]["mid"])

# 時間を確認（デフォルトではNYタイム）
print(r.response["candles"][0]["time"])

# dataとして Pythonのリストへ過去レートを変換
data = []
for raw in r.response['candles']:
    data.append([raw["time"], raw["volume"], raw["mid"]["o"], raw["mid"]["h"], raw["mid"]["l"], raw["mid"]["c"]])

# リストから Pandas DataFrameへ変換
df = pd.DataFrame(data)
df.columns = ["Time", "Volume", "Open", "High", "Low", "Close"]
df = df.set_index("Time")
print(df.head())

# date型を綺麗にする
df.index = pd.to_datetime(df.index)
print(df.tail())

# DataFrame から CSVファイルを書き出し
df.to_csv("api-usdjpy-5s-1115.csv")
