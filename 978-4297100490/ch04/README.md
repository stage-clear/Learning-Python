# 推測統計の基本
一部のデータから全体の統計的性質を推測する枠組みが, **推測統計** です

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%precision 3
%matplotlib inline
```

```python
df = pd.read_csv('../data/ch4_scores400.csv')
scores = np.array(df['点数'])
scores[:10]
```

## 4.1 母集団と標本
- 母集団 *population*
- 標本 *sample*
- 標本抽出 *sampling*
- 推定量 *estimator*
- 推定値 *estimate*

### 4.1.1 標本の抽出方法

> ランダムに標本を抽出する方法を **無作為抽出（random sampling）** といいます

> 複数回同じ標本を選ぶ抽出方法を **復元抽出(sampling with replacement)** 、
> 同じ標本は一度しか選ばない方法を **非復元抽出(sampling without replacement)** といいます。

```python
# 無作為抽出
np.random.choice([1, 2, 3], 3)


# 非復元抽出
np.random.choice([1, 2, 3], 3, replace=False)
```

> 乱数のシードとは, これから発生させる乱数の元となる数字で, これを定めておくと毎回同じ乱数を得ることができます。

```python
np.random.seed(0)
np.random.choice([1, 2, 3], 3)
```

```python
np.random.seed(0)
sample = np.random.choice(scores, 20)
sample.mean()
```

```python
scores.mean()
```

```python
for i in range(5):
    sample = np.random.choice(scores, 20)
    print(f'{i+1}回目の無作為抽出で得た標本平均', sample.mean())
```

## 4.2 確率モデル

> このような不確定さを伴った現象は **確率（probability）** を使って考えることができます。
> 確率を使って無作為抽出やサイコロを数学的にモデル化したものを **確率モデル（probability model）** といいます

### 4.2.1 確率の基本

> サイコロのように, 結果を言い当てることはできないが, とりうる値とその値が出る確率が決まっているものを **確率変数（random variable）** といいます。

> サイコロは投げるまでどの出目が出るかわかりませんが、投げることで出目は１つに確定します。このように確率変数の結果を観察することを **試行（trial）** といい,
> 試行によって観測される値のことを **実現値（realization）** といいます。

> 「出目が１」や「出目が奇数」といった試行の結果起こりうる出来事を **事象（event）** といい、
> 特に「出目が１」といったこれ以上細かく分解できない事象のことを **根元事象（elementary event）** といいます。

> 「事象が互いに排反なら, それらのうち少なくとも１つが起こる事象は, 各事象の確率の和に等しい」
> 事象が **互いに排反** とは, それぞれの事象が同時に起こりえないということです。

### 4.2.2 確率分布

**確率分布（probability distribution）** とは, 確率変数がどのような振る舞いをするかを表したものです.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# いかさまサイコロ
dice = [1, 2, 3, 4, 5, 6]
prob = [1/21, 2/21, 3/21, 4/21, 5/21, 6/21]

np.random.choice(dice, p=prob)

num_trial = 100
sample = np.random.choice(dice, num_trial, p=prob)
sample

freq, _ = np.histogram(sample, bins=6, range=(1, 7))
pd.DataFrame({
    '度数': freq,
    '相対度数': freq / num_trial
}, index=pd.Index(np.arange(1, 7), name='出目'))

// グラフ
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
# 真の確率分布を横線で表示
ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
# 棒グラフの [1.5, 2.5, ... 6.5]の場所に目盛りをつける
ax.set_xticks(np.linspace(1.5, 6.5, 6))
# 目盛りの値は [1, 2, 3, 4, 5, 6]
ax.set_xticklabels(np.arange(1, 7))
ax.set_xlabel('出目')
ax.set_ylabel('相対度数')
plt.show()
```

```python
num_trial = 100000
sample = np.random.choice(dice, size=num_trial, p=prob)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
ax.set_xticks(np.linspace(1.5, 6.5, 6))
ax.set_xticklabels(np.arange(1, 7))
ax.set_xlabel('出目')
ax.set_ylabel('相対度数')
plt.show()
```

## 4.3 推測統計における確率

```python
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(scores, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('点数')
ax.set_ylabel('相対度数')
```

> 無作為抽出においても標本のサンプルサイズを増やしていくと, 標本データの相対度数は実際の確率分布に近づいていきます.

```python
# サンプル数を増やす
sample = np.random.choice(scores, 10000)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample, bins=100, range=(0, 100), density=True)
ax.set_xlim(20, 100)
ax.set_ylim(0, 0.042)
ax.set_xlabel('点数')
ax.set_ylabel('相対度数')
plot.show()
```

```python
# サンプルサイズ20の標本を抽出して標本平均を計算する作業を1000回行う
sample_means = [np.random.choice(scores, 20).mean() for _ in range(10000)]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(sample_means, bins=100, range=(0, 100), density=True)
# 母平均を縦線で表示
ax.vlines(np.mean(scores), 0, 1, 'gray')
ax.set_xlim(50, 90)
ax.set_ylim(0, 0.13)
ax.set_xlabel('点数')
ax.set_ylabel('相対度数')
plt.show()
```

## 4.4 これから学ぶこと

> 推定とは母数を推定することを指し, １つの値で推定することを **点推定** , 区間で推定することを **区間推定** と呼びます.

> 検定とは母集団の統計的性質について仮説をたてて, その仮説が正しいかどうかを判断する手法のことです
