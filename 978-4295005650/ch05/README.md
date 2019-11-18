# 5. 欠損データへの対応
## 5.1 はじめに
欠損値 - missing value

## 5.2 NaN とは何か

```python
# numpy の欠損値だけをインポート
from numpy import NaN, NAN, nan

# 欠損値は他のデータと違って、実際にはどれとも等しくない値である
print(NaN == True) # False
print(NaN == False) # False
print(NaN == 0) # False
print(NaN == '') # False
print(NaN == NaN) # False
print(NaN == nan) # False
print(NaN == NAN) # False
print(nan == NAN) # False
```

```python
import pandas as pd

print(pd.isnull(NaN)) # True
print(pd.isnull(nan)) # True
print(pd.isnull(NAN)) # True

# pandas には欠損値でないことをテストするメソッドもある
print(pd.notnull(NaN)) # False
print(pd.notnull(42)) # True
print(pd.notnull('missing')) # True
```

## 5.3 欠損値はどこから来るのか
データを整える前処理の段階で欠損値が入る場合がある

### 5.3.1 データのロード
read_csv 関数には、欠損地の読み込みに関係する３つのパラメータがある。 

|パラメータ|説明||
|:-|:-|:-|
|na_values|欠損値を指定できる|`na_values=[99]`|
|keep_default_na|デフォルトの欠損値をどう扱うか|`keep_default_na=False`|
|na_filter|欠損値として値を読み込むかどうか|`na_filter=False`|

```python
# データの場所を設定
visited_file = 'data/survey_visited.csv'

# デフォルト値でデータをロード
print(pd.read_csv(visited_file))

# 欠損値のデフォルトなしでデータをロード
print(pd.read_csv(visited_file, keep_default_na=False))

# 欠損値を手作業で指定
print(pd.read_csv(visited_file, na_values=[''], keep_default_na=False))
```

### 5.3.2 マージされたデータ
```python
visited = pd.read_csv('data/survey_visited.csv')
survey = pd.read_csv('data/survey_survey.csv')

print(visited)
print(survey)

vs = visited.merge(survey, left_on='ident', right_on='taken')
print(vs)
```

### 5.3.3 ユーザー入力
```python
# Series に欠損値がある
num_legs = pd.Series({'goat': 4, 'amoeba': nan})

# DataFrame に欠損値がある
scientists = pd.DataFrame({
  'Name': ['Rosaline Franklin', 'William Gosset'],
  'Occupation': ['Chemist', 'Statistician'],
  'Born': ['1920', '1876'],
  'Died': ['1958', '1937'],
  'missing': [NaN, nan]
})
print(scientists)
```

```python
# 新しい DataFrame を作る
scientists = pd.DataFrame({
  'Name': ['Rosaline Frankilin', 'William Gosset'],
  'Occupation': ['Chemist', 'Statistician'],
  'Born': ['1970', '1876'],
  'Died': ['1958', '1937']
})

# 欠損値の列を代入する
scientists['missing'] = nan
print(scientists)
```

### 5.3.4 インデックスの振り直し

```python
gapminder = pd.read_csv('data/gapminder.tsv', sep='\t')

life_exp = gapminder.groupby(['year'])['lifeExp'].mean()

print(life_exp.loc[range(2000, 2010), ])

# 部分集合を抽出
y2000 = life_exp[life_exp.index > 2000]
print(y2000)

# reindex
print(y2000.reindex(range(2000, 2010)))
```

## 5.4 欠損のデータの扱い
### 5.4.1 欠損データを数える

```python
ebola = pd.read_csv('data/country_timeseries.csv')

# 欠損していない値の総数を求める
print(ebola.count())

# 行の総数から欠損のない行数を引く
num_rows = ebola.shape[0]
num_missing = num_rows - ebola.count()
print(num_missing)
```

データに入っている欠損地の総数を知りたいときや、ある特定の列に存在する欠損値を数えたいとき
```python
import numpy as np

print(np.count_nonzero(ebola.isnull()))
print(np.count_nonzero(ebola['Cases_Guinea'].isnull()))
```

```python
# Cases_Guinea の列から値の出現回数を求める
print(ebola.Cases_Guinea.value_counts(dropna=False).head())
```

### 5.4.2 欠損データのクリーニング
#### 5.4.2.1 符号化/置換

```python
print(ebola.fillna(0).iloc[0:15, 0:5])
```

#### 5.4.2.2 前方の値で置換する (Fill Forward)

```python
print(ebola.fillna(method='ffill').iloc[0:10, 0:5])
```

#### 5.4.2.3 後方の値で置換する (Fill Backward)

```python
print(ebola.fillna(method='bfill').iloc[:, 0:5].tail())
```

#### 5.4.2.4 補間する
補間 - interpolation

```python
print(ebola.interpolate().iloc[0:10, 0:5])
```

#### 5.4.2.5 欠損値の削除
```python
print(ebora.shape)

ebola_dropna = ebola.dropna()
print(ebola_dropna.shape)

print(ebola.dropna)
```

###  5.4.3 欠損データとの計算

```python
ebola['Cases_multiple'] = ebola['Cases_Guinea'] + \
                          ebola['Cases_Liberia'] + \
                          ebola['Cases_SierraLeona']
ebola_subset = ebola.loc[:, ['Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone', 'Cases_multiple']]
print(ebola_subset.head(n=10))
```

欠損値を無視する組み込みメソッド: `mean` `sum`

```python
print(ebola.Cases_Guinea.sum(skipna=True))
print(ebola.Cases_Guinea.sum(skipna=False))
```

##  5.5 まとめ
