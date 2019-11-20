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
pandas のメソッドに示していない集約関数も使うことができる。
それには、aggまた aggregateメソッドを呼び出して使いたい集約関数を渡せばよい

#### 10.2.3.1 他のライブラリの関数
```python
import numpy as np

cont_le_agg = df.groupby('continent').lifeExp.agg(np.mean)
print(cont_le_agg)

# agg と aggregate は同じように処理する
cont_le_agg2 = df.groupby('continent').lifeExp.aggregate(np.mean)
print(cont_le_agg2)
```

### 10.2.3.2 カスタムのユーザー関数

```python
def my_mean(values):
    """平均値を計算する自作の関数"""
    # 分母として値の総数を求める
    n = len(values)

    # sum を 0 で初期化
    sum = 0
    for value in values:
        # 値を sum に加算していく
        sum += value

    # sum を値の総数で割った値を返す
    return (sum / n)
```

```python
agg_my_mean = df.groupby('year').lifeExp.agg(my_mean)
print(agg_my_mean)
```

パラメータを複数渡す場合は agg, aggregate に渡すことができる

```python
def my_mean_diff(values, diff_value):
    """平均値とdiff_valueの差を求める"""
    n = len(values)
    sum = 0
    for value in values:
        sum += value
    mean = sum / n
    return (mean - diff_value)

# 全体の平均値を計算
global_mean = df.lifeExp.mean()
print(global_mean)

# 複数のパラメータを持つカスタム集約関数
agg_mean_diff = df.groupby('year').lifeExp.\
    agg(my_mean_diff, diff_value=global_mean)
print(agg_mean_diff)
```

### 10.2.4 複数の関数を同時に計算する

```python
# 年ごとに, lifeExp の度数、平均、標準偏差を計算する
gdf = df.groupby(''year').lifeExp.agg([np.count_nonzero, np.mean, np.std])
print(gdf)
```

### 10.2.5 agg/aggregate で dict を使う
aggにPythonの辞書を渡す

#### 10.2.5.1 DataFrame に対して dict を指定する
```python
# DataFrame オブジェクトに対し, 辞書を使って, 年ごと
# 複数の列を集約し, 余命の平均値, 総人口の中央値
# １人当たりのGDPの中央値を計算する
gdf_dict = df.groupby('year').agg({
    'lifeExp': 'mean',
    'pop': 'median',
    'gdpPercap': 'median'
})

print(gdf_dict)
```

#### 10.2.5.2 Series に対して dict を指定する
```python
gdf = df.groupby('year')['lifeExp'].\
    agg([
        np.count_nonzero,
        np.mean,
        np.std,
    ]).\
    rename(columns={
        'count_nonzero': 'count',
        'mean': 'avg',
        'std': 'std_dev'
    }).\
    reset_index()

print(gdf)
```

## 10.3 変換 (transform)
### 10.3.1 標準スコアの例

```python
def my_zscore(x):
    """与えられたデータのzスコアを計算する
    'x'は値のベクトル（あるいはSeriesオブジェクト）
    """
    return ((x - x.mean()) / x.std())

transform_z = df.groupby('year').lifeExp.transform(my_zscore)

# データの行数に注目
print(df.shape)

# 変換したデータに含まれる値の数に注目
print(transform_z.shape)
```

scipy には独自の zscore関数がある
```python
# scipy.stats から zscore 関数をインポート
from scipy.stats import zscore

# グループごとの zscore を計算
sp_z_grouped = df.groupby('year').lifeExp.trasform(zscore)

# グループ化なしで zscore を計算
sp_z_nogroup = zscore(df.lifeExp)
```

```python
# グループごとの my_zscore
print(transform_z.head())

# グループごとの zscore (scipyの関数)
print(sp_z_grouped.head())

# グループ化なしでの zscore(scipyの関数)
print(sp_z_nogroup[:5])
```

#### 10.3.1.1 欠損値の例

```python
import seaborn as sns
import numpy as np

# 結果を再現できるようにシードを設定する
np.random.seed(42)

# tips からランダムに10行をサンプリングする
tips_10 = sns.load_dataset('tips').sample(10)

# ４個の 'total_bill' 値をランダムに選んで欠損値に変える
tips_10.loc[np.random.permutation(tips_10.index)[:4], 'total_bill'] = np.NaN

print(tips_10)
```

```python
count_sex = tips_10.groupby('sex').count()
print(count_sex)
```

```python
def fill_na_mean(x):
    """与えられたベクトルの平均値で埋める"""
    avg = x.mean()
    return (x.fillna(avg))

# 'sex' ごとに 'total_bill' 平均値を計算
total_bill_group_mean = tips_10.\
    groupby('sex').\
    total_bill.\
    transform(fill_na_mean)

# 元のデータの新しい列に代入する
# 'total_bill' を使えば, 元の列を置換できる
tips_10['fill_total_bill'] = total_bill_group_mean

print(tips_10[['sex', 'total_bill', 'fill_total_bill']])
```

## 10.4 フィルタリング
何らかの真偽値による絞り込み - boolean subsetting

```python
# tips データセットをロードする
tips = sns.load_dataset('tips')

# 元のデータの行数を確認
print(tips.shape)

# テーブルサイズの出現頻度を見る
print(tips['size'].value_counts())
# それぞれのグループに30個以上の観測が含まれるようにしたい

# 各グループが30個以上の観測を持つよう、データにフィルタをかける
tips_filtered = tips.groupby('size').filter(lambda x: x['size'].count() >= 30)

print(tips_filtered.shape)

print(tips_filtered['size'].value_counts())
```

## 10.5 DataFrameGroupBy オブジェクト
https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

### 10.5.1 グループ

```python
tips_10 = sns.load_dataset('tips').sample(10, random_state=42)
print(tips_10)

# グループ化されたオブジェクトだけを保存
grouped = tips_10.groupby('sex')
print(grouped)

# groupby の実際のグループを見ると, インデックスだけが返される
print(grouped.groups)
```

### 10.5.1 複数の変数に関わるグループ計算
Pythonは EAFP の思想に従っている
EAFP(easier to ask for forgiveness than permission)は、「いちいち許可をもらうより、寛容にしてもらうほうが簡単」という考えだ

```python
# 関連する列の平均値を計算する
avgs = grouped.mean()
print(avgs)

# すべての列で平均値が計算されるわけではない
# 全ての列のリスト
print(tips_10.columns)
```

### 1.5.3 グループの抽出

```python
# 'Female' グループを取得
female = grouped.get_group('Female')
print(female)
```

### 10.5.4 グループごとの反復処理

```python
for sex_group in grouped:
    print(sex_group)

# grouped オブジェクトから要素0を取得することはできない
# このオブジェクトは Pythonの本当のコンテナではなく, pandas.core.groupby.DataFrameGroupByなのだ
print(grouped[0])
```

```python
# 最初の要素だけを表示させてみよう
for sex_group in grouped:
    # オブジェクトの型を取得 (tuple)
    print('the type is: {}\n'.format(type(sex_group)))
    
    # オブジェクトの長さを取得 (要素数は2)
    print('the length is: {}\n'.format(len(sex_group)))
    
    # 最初の要素を取得
    first_element = sex_group[0]
    print('the first element is: {}\n'.format(first_element))
    
    # 最初の要素の型
    print('it has a type of: {}\n'.format(type(sex_group[0])))
    
    # 第２の要素を取得
    second_element = sex_group[1]
    print('the second element is: {}\n'.format(second_element))
    
    # 第２の要素の型
    print('it has a type of: {}\n'.format(type(second_element)))
    
    # 今見ているものをプリント
    print('what we have:')
    print(sex_group)
    
    # 一巡で停止
    break
```

### 10.5.5 複数変数のグループ

```python
# sex と time によるグループ化
bill_sex_time = tips_10.groupby(['sex', 'time'])
# 平均値を求める
group_avg = bill_sex_time.mean()
print(group_avg)
```

### 10.5.6 結果を平坦化

```python
# group_avg の型
print(type(group_avg))

print(group_avg.columns)

print(group_avg.index)
# MultiIndex がある。この MultiIndex を使うこともできる
# 平坦な DataFrame が欲しければ reset_index を呼び出すことができる

group_method = tips_10.groupby(['sex', 'time']).mean().reset_index()
print(group_method)

# あるいは、groupby メソッドで as_index=False パラメータを指定してもよい
group_param = tips_10.groupby(['sex', 'time'], as_index=False).mean()
print(group_param)
```

## 10.6 マルチインデックスを使う

```python
# ダウンロードした epi_sim.zip を展開しておく
intv_df = pd.read_csv('data/epi_sim.txt')

# 行数は900万を超える
print(intv_df.shape)

print(intv_df.head())

# ig_type を数える。個々のグループの観測数を取得できる値が必要なのだ
count_only = intv_df.groupby(['rep', 'intervened', 'tr'])['ig_type'].count()
print(count_only)

print(type(count_only))
# groupby は DataFrame を返さない
# <class 'pandas.core.series.Series'>
# マルチインデックスを持つ Series 形式になっている
# 次にまた groupby 演算を行いたければマルチインデックスのレベルを参照する level パラメータを渡す
count_mean = count_only.groupby(level=[0, 1, 2]).mean()
print(count_mean.head())
```

これらの演算を１個のコマンドに組み合わせることができる

```python
count_mean = intv_df.\
    groupby(['rep', 'intervened', 'tr'])['ig_type'].\
    count().\
    groupby(level=[0,1,2]).\
    mean()

import seaborn as sns
import matplotlib.pyplot as plt

fig = sns.lmplot(x='intervened', y='ig_type', hue='rep', col='tr',
                fit_reg=False, data=count_mean.reset_index())
plt.show()
```

```python
import seaborn as sns
import matplotlib.pyplot as plt

cumulative_count = intv_df.\
    groupby(['rep', 'intervened', 'tr'])['ig_type'].\
    count().\
    groupby(level=['rep']).\
    cumsum().\
    reset_index()
    
fig = sns.lmplot(x='intervened', y='ig_type', hue='rep', col='tr',
    fit_reg=False, data=cumulative_count)
plt.show()
```

## 10.7 まとめ
groupby 文は「分割-適用-結合」に従うもの

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
