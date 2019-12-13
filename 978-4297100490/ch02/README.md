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
データから平均を引き、標準偏差で割る操作を **標準化（standardization）** といい、
標準化されたデータを **基準化変量（standardized data）** や **Zスコア（z-score）** といいます。

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

```python
# 50人分の英語の点数の array
english_scores = np.array(df['英語'])
# Series に変換して describe を表示
pd.Series(english_scores).describe()
```

### 2.4.1 度数分布表
分割した区間とデータ数を表にまとめたものが **度数分布表** です。

- **階級（class）**: 区間
- **度数（frequency）**: 各階級に属している数
- **階級幅**: 各区間の幅
- **階級数**: 階級の数

> 度数は np.histogram 関数を使うと簡単に求めることができます。
> bins が階級数を, range で最小値と最大値を指定できます

```python
freq, _ = np.histogram(english_scores, bins=10, range=(0, 100))
freq
```

```python
# 度数分布表
# 0~10, 10~20, ... といった文字列のリストを作る
freq_class = [f'{i}~{i+10}' for i in range(0, 100, 10)]
# freq_class をインデックスにして freq で DataFrame を作る
freq_dist_df = pd.DataFrame({'度数': freq}, index=pd.Index(freq_class, name='階級'))
freq_dist_df
```

**階級値** は各階級を代表する値のことで階級の中央の値が使われます

```python
class_value = [(i + (i + 10))//2 for i in range(0, 100, 10)]
```

**相対度数** は全データ数に対してその階級のデータがどのくらいの割合を占めているかを示します

```python
rel_freq = freq / freq.sum()
```

**累積相対度数** はその階級までの相対度数の輪を示します。<br>
累積和の計算には np.cumsum 関数が便利です

```python
cum_rel_freq = np.cumsum(rel_freq)
cum_rel_freq
```

```python
# 階級値と相対度数と累積相対度数を度数分布表に付け加えます
freq_dist_df['階級値'] = class_value
freq_dist_df['相対度数'] = rel_freq
freq_dist_df['累積相対度数'] = cum_rel_freq
freq_dist_df = freq_dist_df[['階級値', '度数', '相対度数', '累積相対度数']]
```

最頻値
```python
freq_dist_df.loc[freq_dist_df['度数'].idxmax(), '階級値']
```

### 2.4.2 ヒストグラム hist

```python
import matplotlib.pyplot as plt

# グラフが notebook 上に表示されるようにする
%matplotlib inline
```

```python
# キャンバスを作る
# figsize で横・縦の大きさを指定
fig = plt.figure(figsize=(10,6))
# キャンバス上にグラフを描画するための領域を作る
# 引数は領域を1x1個作り, １つめの領域に描画することを意味する
ax = fig.add_subplot(111)

# 階級数を10にしてヒストグラムを描画
freq, _, _ = ax.hist(english_scores, bins=10, range=(0, 100))
# X軸にラベルをつける
ax.set_xlabel('点数')
# Y軸にラベルをつける
ax.set_ylabel('人数')
# X軸に0, 10, 20, ..., 100の目盛りをふる
ax.set_xticks(np.linspace(0, 100, 10+1))
# Y軸に 0, 1, 2, ... の目盛りをふる
ax.set_yticks(np.range(0, freq.max()+1))
# グラフの表示
plt.show()
```

```python
# 階級数を25, 階級幅を4 に指定
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

freq, _, _ = ax.hist(english_scores, bins=25, range=(0, 100))
ax.set_xlabel('点数')
ax.set_ylabel('人数')
ax.set_xticks(np.linspace(0, 100, 25+1))
ax.set_yticks(np.arange(0, freq.max()+1))
plt.show()
```

```python
# 相対度数のヒストグラムを累積相対度数の折れ線グラフと一緒に描画します
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)
# Y軸のスケールが違うグラフをax1お同じ領域上にかけるようにする
ax2 = ax1.twinx()

# 相対度数のヒソトグラムにするためには, 度数データの数で割る必要がある
# これは hist の引数 weight を指定することで実現できる
weights = np.ones_like(english_scores) / len(english_scores)
rel_freq, _, _ = ax1.hist(english_scores, bins=25, range=(0, 100), weights=weights)
cum_rel_freq = np.cumsum(rel_freq)
class_value = [(i + (i + 4))//2 for i in range(0, 100, 4)]
# 折れ線グラフの描画
# 引数 ls を '--' にすることで線が点線に
# 引数 marker を 'o' にすることでデータ点を丸に
# 引数 color を 'gray' にすることで灰色に
ax2.plot(class_value, cum_rel_freq, ls='--', marker='o', color='gray')

# 折れ線グラフの罫線を消去
ax2.grid(visible=False)

ax1.set_xlabel('点数')
ax1.set_ylabel('相対度数')
ax2.set_ylabel('累積相対度数')
ax1.set_xticks(np.linspace(0, 100, 25 + 1))

plt.show()
```

### 2.4.3 箱ひげ図 boxplot
- 箱ひげ図はデータのばらつきを表現するための図です
- 四分位範囲の*Q1*, *Q2*, *Q3*, *Q4*, *IQR* を使います

```python
fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111)
ax.boxplot(english_scores, labels=['英語'])

plt.show()
```




