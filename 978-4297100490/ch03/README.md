# 3. ２次元データの整理


```python
import numpy as np
import pandas as pd

%precision 3
pd.set_option('precision', 3)

df = pd.read_csv('../data/ch2_xcores_em.csv', index_col='生徒番号')
```

```python
en_scores = np.array(df['英語'])[:10]
ma_scores = np.array(df['数学'])[:10]

scores_df = pd.DataFrame({
    '英語': en_scores,
    '数学': ma_scores
}, index=pd.Index(['A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J'], name='生徒'))
```

## 3.1 ２つのデータの関係性の指標
- **正の相関**: 例) 英語の点が高い人ほど数学の点も高い傾向にある
- **負の相関**: 例) 英語の点が高い人ほど数学の点は低い傾向にある
- **無相関**: 例） どちらも当てはまらず、英語の点数が数学の点数に直線的な影響を及ぼさない

### 3.1.1 共分散 covariance

```python
summary_df = scores_df.copy()
summary_df['英語の偏差'] =\
    summary_df['英語'] - summary_df['英語'].mean()
summary_df['数学の偏差'] =\
    summary_df['数学'] - summary_df['数学'].mean()
summary_df['偏差同士の積'] =\
    summary_df['英語の偏差'] * summary_df['数学の偏差']
summary_df
```

```python
summary_df['偏差同士の積'].mean()
```

> 共分散には *S<sub>xy</sub>* という表記がよく使われます<br>
> 今回の場合であれば変数*x*が英語, 変数*y* が数学に対応しています

NumPyの場合, 共分散は cov 関数で求めることができます。
ただし返り値は共分散という値ではなく, **共分散列（covariance matrix）** または **分散共分散行列（variance convariance matrix）**
と呼ばれる行列です。

```python
cov_mat = np.cov(en_scores, ma_scores, ddof=0)
dov_mat

# 1行目と1列目が第１引数の英語
# 2行目と2列目が第２引数の数学
# 1行2列目と2行1列目が英語と数学の共分散に該当します

cov_mat[0, 1], cov_mat[1, 0]
# 共分散
```

```python
# 同じ変数同士の共分散は分散と等しくなっています
cov_mat[0, 0], cov_mat[1, 1]

# 英語と数学の分散（上記と一致する）
np.var(en_scores, ddof=0), np.var(ma_scores, ddof=0)
```

### 3.1.2 相関係数 correlation coefficient

> 単位に依存しない相関を表す指標が求められます.<br>
> 共分散は各データの単位をかけたものになるので, 各データの標準偏差で割ることで単位に依存しない指標を定義できます

相関係数は必ず-1から1の間をとり, データが正の相関をもつほど1に近づき, 負の相関を持つほど-1に, 無相関であれば0になります.

```python
# 英語と数学の点数の相関係数を求める
np.cov(en_scores, ma_scores, ddof=0)[0, 1] /\
    (np.std(en_scores) * np.std(ma_scores))
```

```python
# NumPyの場合
np.corrcoef(en_scores, ma_scores)

# DataFrame の場合
scores_df.corr()
```

## 3.2 ２次元データの視覚化

```python
import matplotlib.pyplot as plt

%matplotlib inline
```

### 3.2.1 散布図

```python
english_scores = np.array(df['英語'])
math_scores = np.array(df['数学'])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# 散布図
ax.scatter(english_scores, math_scores)

ax_set_xlabel('英語')
ax_set_ylabel('数学')

plt.show()
```

### 3.2.2 回帰直線 regression line

```python
# 係数 β_o と β_1 を求める
poly_fit = np.polyfit(english_scores, math_scores, 1)
# β_o + β_1 x を返す関数を作る
poly_1d = np.poly1d(poly_fit)
# 直線を描画するためのx座標を作る
xs = np.linspace(english_scores.min(), english_scores.max())
# xs に対応するy座標を求める
ys = poly_1d(xs)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlabel('英語')
ax.set_ylabel('数学')
ax.scatter(english_scores, math_scores, label='点数')
ax.plot(xs, ys, color='gray', label='f{poly_fit[1]: .2f}+{poly_fit[0]:.2f}x')

# 凡例の表示
ax.plot(xs, ys, color='gray', label='f{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x')
ax.legend(loc='upper left')

plt.show()
```

### 3.2.3 ヒートマップ

```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
c = ax.hist2d(english_scores, math_scores, bins=[9, 8], range=[(35, 80), (55, 95)])
ax.set_xticks(c[1])
ax.set_yticks(c[2])

# カラーバーの表示
fig.colorbar(c[3], ax=ax)

plt.show()
```

## 3.3 アンスコムの例

```python
anscombe_data =np.load('../data/ch3_anscombe.npy')

print(anscombe_data.shape)
anscombe_data[0]
```

```python
stats_df = pd.DataFrame(index=[
    'Xの平均', 'Xの分散', 'Yの分散',
    'Yの分散', 'XとYの相関関係',
    'XとYの回帰直線'
])

for i, data in enumerate(anscombe_data):
    dataX = data[:, 0]
    dataY = data[:, 1]
    poly_fit = np.polyfit(dataX, dataY, 1)
    stats_df[f'data{i+1}'] = \
        [
            f'{np.mean(dataX):.2f}',
            f'{np.var(dataX):.2f}',
            f'{np.mean(dataY):.2f}',
            f'{np.var(dataY):.2f}',
            f'{np.corrcoef(dataX, dataY)[0, 1]:.2f}',
            f'{poly_fit[1]:.2f+{poly_fit[0]:.2f}}x'
        ]

 stats_df
```

```python
# グラフを描画する領域を2x2個作る
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)

xs = np.linspace(0, 30, 100)
for i, data in enumerate(anscombe_data):
    poly_fit = np.polyfit(data[:, 0], data[:, 1], 1)
    poly_id = np.poly1d(poly_fit)
    ys = poly_1d(xs)
    
    # 描画する領域の選択
    ax = axes[i//2, i%2]
    ax.set_xlim([4, 20])
    ax_set_ylim([3, 13])
    # タイトルをつける
    ax.set_title(f'data{i+1}')
    ax.scatter(data[:, 0], data[:,1])
    ax.plot(xs, ys, color='gray')

# グラフ同士の間隔を狭くする
plt.tight_layout()
plot.show()
```








