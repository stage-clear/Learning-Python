# 10. groupby 演算による 分割-適用-結合
## 10.1 はじめに
1. データを、キーによって複数の部分に分割(split)する
2. データの各部に関数を適用(apply)する
3. 各部からの結果を結合(combine)して、新しいデータセットを作る

元のデータを、それぞれ独立した部分に分割して計算を実行できる

groupbyメソッドを使わずに行うこともできる
- 集約(aggregation)は、DataFrame から条件を満たす部分集合を抽出することで実行できる
- 変換(transformation)は、別の関数に列を渡すことによって実行できる
- フィルタリング(filtering)も、条件による抽出で実行できる

**目標**
1. データを集約し、変換し、フィルタリングするための groupby 演算
2. groupby演算を実行するための、組み込み関数とカスタム（自作）のユーザー関数

## 10.2 集約
複数の値を受け取って１個の値を返すプロセス

### 10.2.1 １個の変数で分割する基本的な集約　

```python
import pandas as pd
df = pd.read_csv('data/gapminder.tsv', sep='\t')

avg_life_exp_by_year = df.groupby('year').lifeExp.mean()
# 上記と同義
# avg_life_exp_by_year = df.groupby('year')['lifeExp'].mean()

print(avg_life_exp_by_year)

# データから、ユニークな（重複のない）年のリストを得る
years = df.year.unique()
print(years)

# 1952年のデータを抽出
y1952 = df.loc[df.year == 1952, :]
print(y1952.head())

# 部分集合に対して関数を実行する
y1952_mean = y1952.lifeExp.mean()
print(y1952_mean)
```

### 10.2.2 組み込みの集約メソッド

|pandasのメソッド|numpy/scipyの関数|説明|
|:-|:-|:-|
|count|np.count_nonzero|頻度（NaNの値を含まない）|
|size|--|頻度（NaNの値を含む）|
|mean|np.mean|平均値|
|std|np.std|標本標準偏差|
|min|np.min|最小値|
|quantile(q=0.25)|np.percentile(q=0.25)|25%点の値|
|quantile(q=0.50)|np.percentile(q=0.50)|50%点の値|
|quantile(q=0.75)|np.percentile(q=0.75)|75%点の値|
|max|np.max|最大値|
|sum|np.sum|値の合計|
|var|np.var|不偏分散|
|sem|scipy.stats.sem|平均値の不偏標準誤差|
|describe|scipy.stats.describe|count, mean, std, min, 25%, 50%, 75% max が返す値|
|first|--|最初の行を返す|
|last|--|最後の行を返す|
|nth|--|n番目の行を返す（0から数えて）|

```python
continent_descibe = df.groupby('continent').lifeExp.describe()
print(continent_describe)
```

### 10.2.3 集約関数
