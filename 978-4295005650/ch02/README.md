# pands のデータ構造
## 2.1 はじめに

## 2.2 データを自作する
### 2.2.1 Series を作る
- Seriesオブジェクトはlistと同じく、1次元のコンテナ
- SeriesはDataFrameの各列を表現するデータ型

```pyhton
import pandas as pd

s = pd.Series(['banana', 42])
print(s)

# 手作業で Series にインデックスを代入するには
# Pythonのlistを渡す
s = pd.Series(['Wes McKinney', 'Creator of pandas'], index=['Person', 'Who'])
print(s)
```

### 2.2.2 DataFrame を作る
```python
scientists = pd.DataFrame({
  'Name': ['Rosaline Franklin', 'William Gosset'],
  'Occupation': ['Chemist', 'Statistician'],
  'Born': ['1920', '1876'],
  'Died': ['1958', '1937'],
  'Age': [37, 61]
})

print(scientists)
```

順序を指定する
```python
scientists = pd.DataFrame(
  data={
    'Occupation': ['Chemist', 'Statistician'],
    'Born': ['1920', '1876'],
    'Died': ['1958', '1937'],
    'Age': [37, 61]
  },
  index=['Rosaline Franklin', 'William Gosset'],
  columns=['Occupation', 'Born', 'Age', 'Died', ]
)

print(scientists)
```

```python
from collections import OrderedDict

# OrderedDict で丸カッコを使っている箇所に注目
# 丸カッコに入れて「２個のタプルのリスト」を渡している
scientists =pd.DataFrame(OrderedDict([
  ('Name', ['Rosaline Franklin', 'William Gosset']),
  ('Occupation', ['Chemist', 'Statisitician']),
  ('Born', ['1920', '1876']),
  ('Died', ['1958', '1937']),
  ('Age', [37, 61])
]))

print(scientists)
```

## 2.3 Series について

```python
scientists = pd.DataFrame(
  data={
    'Occupation': ['Chemist', 'Statistician'],
    'Born': ['1920', '1876'],
    'Died': ['1958', '1937'],
    'Age': [37, 61]
  },
  index=['Rosaline Franklin', 'William Gosset'],
  columns=['Occupation', 'Born', 'Died', 'Age']
)
print(scientists)
```

```python
# 行インデックスラベルによる選択
first_row = scientists.loc['William Gosset']

print(type(first_row))
print(first_row)
print(first_row.index)
print(first_row.values)
print(first_row.keys())

# 属性を使って最初のインデックスをとる
print(first_row.index[0])

# メソッドを使って最初のインデックスをとる
print(first_row.keys()[0])
```
**Seriesで使える属性**
|Seriesの属性|説明|
|:-|:-|
|loc|インデックスの値を使った絞り込み（サブセットの抽出）|
|iloc|インデックスの位置を使った絞り込み（サブセットの抽出）|
|ix|インデックスの値・位置を使った絞り込み（サブセットの抽出）|
|dtype または dtypes|Seriesの内容の型|
|T|Sereis の転置（Transpose）|
|Shape|データの行数と列数|
|size|Seriesにある全要素の数|
|values|Seriesのndarray（あるいは、それに似たもの）|

### 2.3.1 Seriesはndarrayに似たもの
#### Series のメソッド

```python
# Ageの列をとる
ages = scientists['Age']
print(ages)
```

Seriesはnumpy.ndarrayの拡張と考えられるものなので、共通した属性やメソッドがある
```python
print(ages.mean())
print(ages.min())
print(ages.max())
print(ages.std())
```

**Seriesに対して実行できるメソッド（一部）**
|Seriesメソッド|説明|
|:-|:-|
|append|２つ以上のSeriesを連結する|
|corr|もう１つのSeriesとの相関を計算する|
|cov|もう１つのSeriesとの共分散を計算する|
|describe|要約統計量を計算する|
|drop_duplicates|重複のないSeriesを返す|
|equals|Seriesに含まれる要素が同じか判定する|
|get_values|Seriesの値をとる（values属性と同じ）|
|hist|ヒストグラムを描画する|
|isin|値がSeriesに含まれているかチェックする|
|min|最小値を返す|
|max|最大値を返す|
|mean|算術平均を返す|
|median|中央値を返す|
|mode|最頻値を返す|
|quantile|所与の分位点の値を返す|
|replace|Seriesに含まれる値を指定された値で置き換える|
|sample|Seriesの値から無作為標本を１つ返す|
|sort_values|値をソートする|
|to_frame|SeriesをDataFrameに変換する|
|transpose|転置を返す|
|unique|ユニークな値のnumpy.ndarrayを返す|

### 2.3.2 真偽値による絞り込み
```python
scientists = pd.read_csv('scientists.csv')

ages = scientists['Age']
print(ages)

# 基本的な記述統計を行う
print(ages.describe())

# すべての年齢の平均値
print(ages.mean())

# 平均値を上回る年齢だけに絞り込む
print(ages[ages > ages.mean()])

print(ages > ages.mean())
print(type(ages > ages.mean()))
```

真偽値のベクトルを手作業で提供することによって、データを絞り込むことも可能
```python
manual_bool_values = [True, True, False, False, True, True, False, True]
print(ages[manual_bool_values])
```

### 2.3.3 演算の自動的な整列とベクトル化（ブロードキャスティング）
#### 2.3.3.1 同じ長さのベクトル
```python
print(ages + ages)

print(ages * ages)
```

#### 2.3.3.2 ベクトルと整数値（スカラー）
```python
print(ages + 100)

print(ages * 2)
```

#### 2.3.3.3 長さの違うベクトル
```python
print(ages + pd.Series([1,100]))
# 残りの部分は欠損値（NaN）で埋められる
```

```python
# 他の型との演算では、データ数が同じでなければならない
import numpy as np

# これはエラーになる
print(ages + np.array([1, 100]))
```

#### 2.3.3.4 インデックスラベルが共通するベクトル（自動整列）

```python
print(ages)

rev_ages = ages.sort_index(ascending=False)
print(ages)

print(ages * 2)

# たとえベクトルの１つが逆順でも
# 上の例と同じ出力が得られる
print(ages + rev_ages)
```

## 2.4 DataFrame について
### 2.4.1 真偽値による絞り込み: DataFrame
```python
# 真偽値のベクトルによる
# 行の絞り込みが行われる
print(scientists[scientists['Age'] > scientists['Age'].mean()])
```

**DataFrameのサブセットを抽出する方法**
|構文|選択結果|
|:-|:-|
|df[列の名前]|1列|
|df[[列1, 列2, ...]]|複数の列|
|df.loc[行のラベル]|インデックスラベル（行の名前）による1行|
|df.loc[[ラベル1, ラベル2,...]]|インデックスラベルによる複数行|
|df.iloc[行番号]|行番号による1行|
|df.iloc[行1, 行2,...]|行番号による複数行|
|df[bool]|真偽値よる１行|
|df[[bool1, bool2,...]]|真偽値よる複数行|
|df[start:stop:step]|スライス構文による複数行|

### 2.4.2 演算による整列とベクトル化（ブロードキャスティング）
```python
first_half = scientists[:4]
second_half = scientists[4:]
```

```python
# １個のスカラーによる乗算
print(scientists * 2)
```

## 2.5 Series と DataFrame の書き換え
### 2.5.2 列を追加する
```python
print(scientists['Born'].dtype)
# -> object

print(scientists['Died'].dtype)
# -> object

# 'Born' 列をdatetime型に変換する
born_datetime = pd.to_datetime(scientists['Born'], format='%Y-%m-%d')
print(born_datetime)

# 'Died'列をdatetime型に変換する
died_datetime = pd.to_datetime(scientists['Died'], format='%Y-%m-%d')
print(died_datetime)

# object（文字列）の日付を datetime で表現した上で新しい列の集合を作る
scientists['born_dt'], scientists['died_dt'] = (born_datetime, died_datetime)
print(scientists.head())
```

### 2.5.2 列を直接変更する
```python
print(scientists['Age'])
```

```python
import random

# シードを設定（常に同じ擬似乱数が得られる）
random.seed(42)
random.shuffle(scientists['Age'])

# random_state を使ってランダム化を弱める
scientists['Age'] = scientists['Age'].\
  sample(len(scientists['Age']), random_state=24).\
  reset_index(drop=True) # 年齢の値だけがランダム化される

# この列は２回シャッフルしている
print(scientists['Age'])
```

```python
# 日付の引き算によって日数をとる
scientists['age_days_dt'] = (scientists['died_dt'] - \
  scientists['born_dt'])
print(scientists)

# 日数の値を年数だけに変換するには astype メソッドを使う
scientists['age_years_dt'] = scientists['age_days_dt'].\
  astype('timedelta64[Y]')
print(scintists)
```

### 2.5.3 列を捨てる
```python
# データに現在ある全部の列を確認する
print(scientists.columns)

# シャッフルしたAgeの列を捨てる
# 列を捨てるには axis=1 という引数を渡す
scientists_dropped = scientists.drop(['Age'], axis=1)

# Age 列を捨てた後の列
print(scientists.columns)
```

## 2.6 データのエクスポートとインポート
### 2.6.1 pickle
データ（オブジェクト）をシリアライズし、バイナリフォーマットで保存する

#### Series
```python
names = scientists['Name']
print(names)

# 保存先へのパスを示す文字列を渡す
names.to_pickle('./output/scientists_names_series.pickle')
```

#### 2.6.1.2 DataFrame
```python
scientists.to_pickle('output/scientists_df.pickle')
```

#### 2.6.1.3 pickle データを読む
```python
# Seriesの場合
scientist_names_from_pickle = pd.read_pickle('output/scientists_name_series.pickle')
print(scientist_names_from_pickle)

# DataFrameの場合
scientists_from_pickle = pd.read_pickle('output/scientists_df.pickle')
print(scientists_from_pickle)
```

### 2.6.2 CSV
```python
# SeriesオブジェクトをCSVに保存する
names.to_csv('output/scientist_names_series.csv')

# DataFrameオブジェクトをTSVに保存する
scientists.to_csv('output/scientists_df.tsv', sep='\t')
```

#### 2.6.2.2 CSVデータをインポートする
```
pd.read_csv()
```

### 2.6.3 Excel
明示的な `to_excel` メソッドはない

#### Series
```python
# Excelファイルを保存するために
# まず Series を DataFrame に変換する
names_df = names.to_frame()

import xlwt # インストールの必要あり
# xls ファイル
names_def.to_excel('output/scientists_names_series.xls')

import openpyxl # インストールの必要あり
# 新しい xlsx ファイル
names_df.to_excel('output/scientists_namesseries_df.xlsx')
```

#### 2.6.3.2 DataFrame
```python
# DataFrameをExcelフォーマットで保存
scientists.to_excel('output/scientists_df.xlsx', sheet_name='scientists', index=False)
```

### feather フォーマット: R言語とのインターフェイス

### その他のデータ出力形式
|メソッド|説明|
|:-|:-|
|to_clipboard|システムのクリップボードにデータを保存してペースト可能にする|
|to_dense|データを正規（dense）の DataFrame に変換する|
|to_dict|データをPythonの dict に変換する|
|to_gbq|データをGoogle BigQueryの表に変換する|
|to_hdf|データをHDFで保存する|
|to_msgpack|データをポータブルなJSONライクのバイナリに保存する|
|to_html|データをHTMLの表に保存する|
|to_json|データをJSON文字列に変換する|
|to_latex|データをLATEXの表環境用に変換する|
|to_records|データをレコード配列に変換する|
|to_string|DateFrameをstdout表示用の文字列に変換する|
|to_sparse|データをSparseDataFrameに変換する|
|to_sql|データをSQLデータベースに保存する|
|to_stata|データをStataのdtaファイルに変換する|
