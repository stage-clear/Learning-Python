# 6. 整然データを作る
## 6.1 はじめに
- 各行が１回の観測である
- 各行が１個の変数である
- 観察単位の型ごとに、１個の表が構成されている

**目標**
1. 列を行に変える `unpivot` `melt` `gather`
2. 行を列に変える `pivot` `cast` `spread`
3. データを正規化するため、DataFrame を複数の表に分割する
4. 複数のパートからデータを集める

## 6.2 複数列に（変数ではなく）値が入っているとき
### 6.2.1 1列に集める

```python
import pandas as pd
pew = pd.read_csv('data/pew.csv')

# はじめの6列だけを示す
print(pew.iloc[:, 0:6])

# region 列を除くすべての列を融解するので
# value_vars を指定する必要はない
pew_long = pd.melt(pew, id_vars='religion')

print(pew_long.head())

print(pew_long.tail())
```

```python
pew_long = pd.melt(pew,
  id_vars='religion',
  var_name='income',
  value_name='count'
)

print(pew_long.head())

print(pew_long.tail())
```

### 6.2.2 複数の列を残す

```python
billboard = pd.read_csv('data/billboard.csv')

# 行および列の先頭部分
print(billboard.iloc[0:5, 0:16])
```

```python
billboard_long = pd.melt(
  billboard,
  id_vars=['year', 'artist', 'track', 'time', 'date.enterd'],
  var_name='week',
  value_name='rating'
)

print(billboard_long.head())

print(billboard_long.tail())
```

## 6.3 複数の変数を含む列がある場合

```python
ebola = pd.read_csv('data/country_timeseries.csv')
print(ebola.columns)

# 選んだ列を出力
print(ebola.iloc[:5, [0, 1, 2, 3, 10, 11]])
```

**Cases_Guinea** と **Deaths_Guinea** はそれぞれ実際には２つの変数を含んでいる
個別の状態である **Cases** （患者の数）と **Deaths** （死者の数）、そして国名の **Guinea** だ。

```python
ebola_long = pd.melt(ebola, id_vars=['Date', 'Day'])
print(ebola_long.head())

print(ebola_long.tail())
```

### 6.3.1 列を分割して追加する単純な方法
```python
# 変数の列を取得し、文字列メソッドにアクセスすることで、
# デリ見たによる列の分割を行う
variable_split = ebola_long.variable.str.split('_')

print(variable_split[:5])

print(variable_split[-5:])

# コンテナ全体
print(type(variable_split))

# コンテナにある最初の要素
print(type(variable_split[0]))
```

```python
status_values = variable_split.str.get(0)
country_values = variable_split.str.get(1)

print(status_values[:5])

print(status_values[-5:])

print(country_values[:5])

print(country_values[-5:])

ebola_long['status'] = status_values
ebola_long['country'] = country_values

print(ebola_long.head())
```

### 6.3.2 分割と結合を一度に行う（単純な方法）
- 元のデータと同じ順序でベクトルが返されるという事実を利用する
- split の結果が２つの要素のリストとして返されるという事実を利用する

```python
constants = ['pi', 'e']
values = ['3.14', '2.718']
```

```python
# zip オブジェクトの内容を表示するには、
# zip 関数に対して list を呼び出す必要がある
# Python3 では zip は イテレータを返す
print(list(zip(constants, values)))

ebola_long['status'], ebola_long['country'] = \
  zip(*ebola_long.variable.str.split('_'))
print(ebola_long.head())
```

## 6.4 行と列の両方に変数があるとき
```python
weather = pd.read_csv('data_weather.csv')
print(weather.iloc[:5, :11])

weather_melt =  pd.melt(weather,
  id_vars=['id', 'year', 'month', 'element'],
  var_name='day',
  value_name='temp'
)

print(weather_melt.head())

print(weather_melt.tail())
```

```python
weather_tidy = weather_melt.pivot_table(
  index=['id', 'year', 'month', 'day'],
  columns='element',
  values='temp'
)

weather_tidy_flat = weather_tidy.reset_index()
print(weather_tidy_flat_head())

weather_tidy = weather_melt.\
  pivot_table(
    index=['id', 'year', 'month', 'day'],
    columns='element',
    values='temp'
  ).\
  reset_index()

print(weather_tidy.head())
```

## 6.5 １個の表に観察単位が複数あるとき（正規化）
```python
billboard_songs = billboard_long[['year', 'artist', 'track', 'time']]
print(billboard_songs.shape)

# 重複する行を削除する
billboard_songs = billboard_songs.drop_duplicates()
print(billboard_songs.shape)

# データの各行にユニークな値を割り当てる
billboard_songs['id'] = range(len(billboard_songs))
print(billboard_songs.head(n=10))

# 曲の DataFrame オブジェクトを元のデータセットにマージ
billboard_ratings = billboard_long.merge(
  billboard_songs, on=['year', 'artist', 'track', 'time'])

print(billboard_ratings.shape)

print(billboard_ratings.head())

billboard_ratings = \
  billboard_ratings[['id', 'date.enterd', 'week', 'rating']]
print(billboard_ratings.head())
```

## 6.6 同じ観察単位が複数の表にまたがっているとき

```python
import os
import urllib

# データをダウンロードするためのコード
# ファイルリストの先頭から、
# ５つのデータセットだけをダウンロードする
with open('data/raw_data_urls.txt', 'r') as data_urls:
    for line, url in enumerate(data_urls):
        if line == 5:
            break
         fn = url.split('/')[-1].strip()
         fp = os.path.join('..', 'data', fn)
         print(url)
         print(fp)
         urllib.request.urlretrieve(url, fp)
```

```python
import glob
# 指定したフォルダから csv ファイルのリストを取得
nyc_taxi_data = glob.glob('../data/fhv_*')
print(nyc_taxi_data)
# これでロードしたいファイル名のリストが手に入った
```

```python
taxi1 = pd.read_csv(nyc_taxi_data[0])
taxi2 = pd.read_csv(nyc_taxi_data[1])
taxi3 = pd.read_csv(nyc_taxi_data[2])
taxi4 = pd.read_csv(nyc_taxi_data[3])
taxi5 = pd.read_csv(nyc_taxi_data[4])

print(taxi1.head(n=2))
print(taxi2.head(n=2))
print(taxi3.head(n=2))
print(taxi4.head(n=2))
print(taxi5.head(n=2))

# それぞれの DataFrame オブジェクトの形状
print(taxi1.shape)
print(taxi2.shape)
print(taxi3.shape)
print(taxi4.shape)
print(taxi5.shape)

# DataFrame を連結する
taxi = pd.concat([taxi1, taxi2, taxi3, taxi4, taxi5])

# 連結したタクシーデータの形状
print(taxi.shape)
```

### 6.6.1 ループを使って複数のファイルをロードする
```python
# 追加するために、最初は空のリストを作る
list_taxi_df = []

# 個々のCSVファイルを反復処理
for csv_filename in nyc_taxi_data:
    # デバッグ用にファイル名をプリントできる
    print(csv_filename)
    
    # CSVファイルを１個の DataFrame オブジェクトにロード
    df = pd.read_csv(csv_filename)
    
    # DataFrameをリストに追加する
    list_taxi_df.append(df)

# DataFrame のリストの長さ
print(len(list_taxi_df))

# 最初の要素の型
print(type(list_taxi_df[0]))

# 最初の DataFrameの先頭を見る
print(list_taxi_df0].head())

taxi_loop_concat = pd.concat(list_taxi_df)
print(taxi_loop_concat.shape)

# 手作業でロードして連結したのと同じ結果か？
print(taxi.equals(taxi_loop_concat)) # True
```

### 6.6.2 リスト内包処理を使って複数のファイルをロードする
リスト内包処理 - comprehension

```python
# ループのコード（コメントなし）
list_taxi_df = []
for csv_filename in nyc_taxi_data:
    df = pd.read_csv(csv_filename)
    list_taxi_df.append(df)

# リスト内包処理を使った同じコード
list_taxi_df_comp = [pd.read_csv(data) for data in nyc_taxi_data]

print(type(list_taxi_df_comp))

# 連結する
taxi_loop_concat_compo = pd.concat(list_taxi_df_comp)

# 連結された DataFrame はどちらも同じか？
print(taxi_loop_concat_comp.equals(taxi_loop_concat))
```

## 6.7 まとめ
