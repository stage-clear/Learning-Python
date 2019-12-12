# 2. １次元データの整理

データの特徴を掴むための方法
1. 平均や分散といった数値の指標によってデータを要約する
2. 図示することで視覚的にデータを俯瞰する

```python
import numpy as np
import pandas as pd

pd.set_option('precision', 3)

df = pd.read_csv('../data/ch2_scores_em.csv', index_col='生徒番号')

df.head()
```

```python
scores = np.array(df['英語'])[:10]

scores
```

```python
scores_df = pd.DataFrame({'点数': scores}, 
  index=pd.Index([
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J'
  ], name='生徒'))
```

## 2.1 データの中心の指標
### 2.1.1 平均値 mean
**平均値**はデータをすべて足し合わせて、データの数で割ることで求まります

```python
# Python
sum(scores) / len(scores)

# NumPy
np.mean(scores)

# Pandas
scores_df.mean()
```

### 2.1.2 中央値 median
**中央値**はデータを大きさの順に並べた時にちょうど中央に位置する値というものでした

- データ数 *n* が奇数なら, `(n + 1) / 2` 番目のデータが中央値
- データ数 *n* が偶数なら, `n / 2` 番目のデータと `n / 2 + 1` 番目のデータの平均が中央値

```python
sorted_scores = np.sort(scores)
sorted_scores

n = len(sorted_scores)

if n % 2 == 0:
  m0 = sorted_scores[n//2 - 1]
  m1 = sorted_scores[n//2]
else:
  median = sorted_scores[(n + 1)//2 - 1]

median
```

```python
# NumPy
np.median(scores)

# pandas
scores_df.median()
```

### 2.1.3 最頻値 mode
**最頻値**はデータの中で最も多く出現する値のことです

```python
pd.Series([1,1,1,2,2,3]).mode()
```

## 2.2 データのばらつきの指標
### 2.2.1 分散と標準偏差
#### 偏差 deviation
**偏差** は各データが平均からどれだけ離れているかを表す指標です

```python
mean = np.mean(scores)
deviation = scores - mean
deviation
```

```python
another_scores = [50, 60, 58, 54, 51, 56, 57, 53, 52, 59]
another_mean = np.mean(another_scores)
another_deviation = another_scores - another_mean
another_deviation
```

> 偏差の平均は常に０になります

```python
np.mean(deviation) # -> 0
np.mean(another_deviation) # -> 0
```

#### 分散 variance

```python
np.mean(deviation ** 2)

# NumPy では var という関数で計算できます
np.var(scores)

# pandas
# pandasのvarメソッドでは違う値になってしまう（不偏分散による）
scores_df.var() # 不偏分散
scores_df.var(ddof=0) #標本分散
```

```python
summary_df['偏差二乗'] = np.square(deviation)
summary_df

summary_df.mean()
```

分散の別のイメージとしては面積の平均という考え方もあります

#### 標準偏差 standard deviation
分散のルートをとったばらつきの指標

```python
np.sqrt(np.var(scores, ddof=0))
np.std(scores, ddof=0) # std関数でも同様
```

### 2.2.2 範囲と四分位範囲
#### 範囲 range
範囲は分散や標準偏差とは異なり、データ全体を見るのではなく、データの最大値と最小値だけでばらつきを表現する方法です

```python
np.max(scores) - np.min(scores)
```

#### 四分位範囲 interquartile range

```latex
# 公式
IQR = Q3 - Q1
```

```python
# 実装
scores_Q1 = np.percentile(scores, 25)
scores_Q3 = np.percentile(scores, 75)
scores_IQR = scores_Q3 - scores_Q1
scores_IQR
```
Q2は中央値に一致します

> 分散は**平均**に対して定義されるばらつきの指標でしたが、*IQR*は**中央値**に対して定義されるばらつきの指標と解釈できます

### 2.2.3 データの指標のまとめ
> DataFrameやSeriesには describe という、ここまで扱ってきたさまざまな指標を一度に求めることができる便利なメソッドがあります

```python
pd.Series(scores).describe()
```

## 2.3 データの正規化 normalization
データを統一的な指標に変換することを正規化といいます

### 2.3.1 標準化 standardization
データから平均を引き、標準偏差で割る操作を**標準化（standardization）**といい、
標準化されたデータを**基準化変量（standardized data）**や**Zスコア（z-score）**といいます。

```python
z = (scores - np.mean(scores)) / np.std(scores)
```

標準化されたデータは平均が０で標準偏差が１になります

```python
np.mean(z), np.std(z, ddof=0)
```

### 2.3.2 偏差値
**偏差値**は平均が50、標準偏差が10になるように正規化した値のことをいいます。

```python
z = 50 + 10 * (scores - np.mean(scores)) / np.std(scores)
```

```python
scores_df['偏差値'] = z
scores_df
```

## 2.4 １次元データの視覚化
