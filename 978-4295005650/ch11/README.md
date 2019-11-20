# 11. 日付/時刻データの操作
## 11.1 はじめに
**目標**
1. Python 組み込みの datetime ライブラリ
2. 文字列を日時に変換する
3. 日時のフォーマット
4. 日付の各部を抽出する
5. 日時の計算を実行する
6. DataFrame にある日時処理
7. リサンプリング（再標本化）
8. 時間帯の処理

## 11.2 Python の datetime オブジェクト

```python
from datetime import datetime
```

```python
# 現在の日付と時刻
now = datetime.now()
print(now)

# 手作業で作る
t1 = datetime.now()
t2 = datetime(1970, 1, 1)
diff = t1 - t2
print(diff)
# <class 'datetime.timedelta'>
# この引き算の結果は timedelta型となる
```

## 11.3 datetime への変換
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

```python
import pandas as pd

ebola = pd.read_csv('data/country_timeseries.csv')

# 元データの左上隅を見る
# Date列に日付が含まれているが、infoを調べてみると
# 実際には pandas の汎用文字列オブジェクトである
print(ebola.iloc[:5, :5])

print(ebola.info())

ebola['date_dt'] = pd.to_datetime(ebola['Data'])

# より明示的に書くと,
ebola['date_dt'] = pd.to_datetime(ebola['Date'], format='%m/%d/%Y')

print(ebola.info())
```

[Pythonの「strftime() と strptime() の振る舞い」](https://docs.python.org/ja/3/library/datetime.html#strftime-and-strptime-format-codes)

## 11.4 日付を含むデータをロードする
read_csv 関数には数多くのパラメータがあるが、日付に関するパラメータは、
- `parse_dates`
- `inher_datetime_format`
- `keep_date_col`
- `date_parser`
- `dayfirst`
などだ。

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

```python
# `parse_dates` で列を指定する
ebola = pd.read_csv('data/country_timeseries.csv', parse_dates=[0])
print(ebola.info())
```

## 11.5 日付のコンポーネントを抽出する

```python
d = pd.to_datetime('2016-02-29')
print(d)
print(type(d))
# <class 'pandas._libs.tslib.Timestamp'>

print(d.year)

print(d.month)

print(d.day)
```

```python
ebola['date_dt'] = pd.to_datetime(ebola['Date'])

print(ebola[['Date', 'date_dt']].head())

ebola['year'] = ebola['date_dt'].dt.year
print(ebola[['Date', 'date_dt', 'year']].head())

ebola['month'], ebola['day'] = (
  ebola['date_dt'].dt.month,
  ebola['date_dt'].dt.day
)

print(ebola[['Date' 'date_dt', 'year', 'month', 'day']].head())

print(ebola.info())
```

## 11.6 日付の計算と timedelta

```python
print(ebola.iloc[-5:, :5])

# 最初の日付を取得
print(ebola['data_dt'].min())

ebola['outbreak_d'] = ebola['data_dt'] - ebola['date_dt'].min()

print(ebola[['Date', 'Day', 'outbreak_d']].head())

print(ebola[['Date', 'Day', 'outbreak_d']].tail())

print(ebola.info())
```

## 11.7 datetime のメソッド

```python
banks = pd.read_csv('data/banklist.csv')
print(banks.head())
```

```python
# インポートするときに、日付を直接解析できる
banks = pd.read_csv('data/banklist.csv', parse_dates=[5, 6])
print(banks.info())

# 銀行が閉鎖した四半期と年を取り出す
banks['closing_quarter'], banks['closing_year'] = \
  (banks['Closing Date'].dt.quarter,
   banks['Closing Date'].dt.year)

# それぞれの年にいくつ銀行が閉鎖したかを計算する
closing_year = banks.groupby(['closing_year']).size()

# 各年の各四半期にいくつ銀行が閉鎖したか
closing_year_q = banks.groupby(['closing_year', 'closing_quarter']).size()
```

```python
# 結果をプロットする
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax = closing_year.plot()
plt.show()

fig, ax = plt.subplots()
ax = closing_year_q.plot()
plt.show()
```

## 11.8 株価データを取得する

```python
# pandas_datareader をインストールして利用できる
import pandas_datareader as pdr

# この例では Tesla に関する株価情報を取得する
tesla = pdr.get_data_yahoo('TSLA')

# 株価データは保存・変更しておき、今後はインターネットに依存せず、
# 同じデータセットをファイルとしてロードする
# tesla.to_csv('data/tesla_stock_yahoo.csv')
tesla = pd.read_csv('data/tesla_stock_yahoo.csv', parse_dates=[0])
```

## 11.9 日付による絞り込み

```python
# 2010年6月のデータだけが欲しい場合
print(tesla.loc[(tesla.Date.dt.year == 2015) & (tesla.Date.dt.month == 6)])
```

### 11.9.1 DatetimeIndex オブジェクト
datetimeオブジェクトをDataFrameオブジェクトのインデックスに設定したい場合

```python
tesla.index = tesla['Date']
print(desla.index)
print(tesla['2015'].iloc[:5, :5]) # ５行-５列
```

```python
# 年と月によってデータを絞り込む
print(tesla['2015-06'].iloc[:, :5]) # 全ての行-5列
```

### 11.9.2 TimedeltaIndex オブジェクト
timedeltaでインデックスを設定する場合

```python
# timedelta を作成する
tesla['ref_date'] = tesla['Date'] - tesla['Date'].min()

tesla.index = tesla['ref_date']

print(tesla.iloc[:5, :5])

# これらの増分（delta）をもとに、データを選択できる
print(tesla['0 day': '5 day'].iloc[:5, :5])
```

## 11.10 日付の範囲
データに日付が欠けていることはよくある

```python
ebola = pd.read_csv('data/country_timeseries.csv', parse_dates=[0])
print(ebola.iloc[:5, :5])
```

reindexメソッドによってインデックスを作り直すために日付の範囲を作るのは、よく行われる

```python
head_range = pd.date_range(start='2014-12-31', end='2015-01-05')

print(head_range)

# 最初の５行だけ処理する
ebola_5 = ebola.head()

# Date をインデックスとして設定する
ebola_5.index = ebola_5['Date']

# それから reindex メソッドを呼び出す

ebola_5.reindex(head_range)
print(ebola_5.iloc[:, :5]) # 全ての行-5列
```


