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
