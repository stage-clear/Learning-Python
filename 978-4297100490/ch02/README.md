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


