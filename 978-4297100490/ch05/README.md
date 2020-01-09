# 離散型確率変数

```python
import numpy as np
import matplotlib.pyplot as plt

%precision 3
%matplotlib inline
```

## 5.1 １次元の離散型確率変数

**離散型確率変数** はとりうる値が離散的な確率変数のことです.

### 5.1.1 １次元の離散型確率変数の定義
#### 確率質量関数

```mathematica
P(X = xk) = pk (k = 1,2,...)

f(x) = P(X = x)
```

となる関数 *f(x)* を **確率質量関数（probability mass function, PMF）** , または **確率関数** と呼びます.

```python
x_set = np.array([1, 2, 3, 4, 5, 6])
```

```mathematica
f(x) = x/21 (x ∈ {1, 2, 3, 4, 5, 6})
       0    (otherwise)
```

```python
// 実装
def f(x):
    if x in x_set:
        return x / 21
    else:
        return 0
```

```python
X = [x_set, f]
```

```python
# 確率 p_k を求める
prob = np.array([f(x_k) for x_k in x_set])
# x_k と p_k の対応を辞書型にして表示
dict(zip(x_set, prob))
```

```python
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.bar(x_set, prob)
ax.set_xlabel('とりうる値')
ax.set_ylabel('確率')

plt.show()
```

#### 確率の性質

> 確率は絶対に0以上で, すべての確率を足すと１にならなければなりません

```python
# np.all はすべての要素が真のときのみ真を返す関数
np.all(prob >= 0)

# 確率の総和が１になっていることを確認
np.sum(prob)
```

#### 累積分布関数
> 確率変数 *X* が *x* 以下になるときの確率を返す関数もよく使われます。
> そのような関数 *F(x)* を **累積分布関数（cumulative distribution function, CDF）** , または単に分布関数と呼びます。

```mathematica
F(x) = P(X < x) = Σ f(xk)
                 xk<x
```

```python
def F(x):
    return np.sum([f(x_k) for x_k in x_set if x_k <= x])

// 出目が３以下になる確率は次のように求めます
F(3)
```

#### 確率変数の変換
確率変数の変換とは, 確率変数 *X* に２をかけて３を足した *2X + 3* といったもので,
確率変数を標準化する（平均を引いて標準偏差で割る）ときなどに重要となる操作です.

```python
y_set = np.array([2 * x_k + 3 for x_k in x_set])
prob = np.array([f(x_k) for x_k in x_set])
dict(zip(y_set, prob))
```

### 5.1.2 １次元の離散型確率変数の指標

#### 期待値
> 確率変数の平均とは確率変数を何回も（無限回）試行して得られた実現値の平均のことを指します

> 残念ながら無限回の試行を行うことはできません.
> そのため離散型変数の場合, **確率変数の平均は確率変数のとりうる値とその確率の積の総和** として定義されます

```mathematica
E(X) = Σ xkf(xk)
       k
```

> 確率変数の平均は **期待値（expected value）** とも呼ばれます<br>
> 記号には *μ* や *E(X)* という表記がよく使われます

```python
# 期待値を計算
np.sum([x_k * f(x_k) for x_k in x_set])

# 100万（10の6乗）回サイコロを振る
sample = np.random.choice(x_set, int(1e6), p=prob)
np.mean(sample)
```

```mathematica
# 離散型確率変数の期待値
E(g(x)) = Σ g(xk)f(xk)
          k
# * 引数gが確率変数に対する変換の関数担っています
```

```python
# 実装
def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])

# gに何も指定しなければ確率変数Xの期待値を求めることになります
```

```python
E(X, g=lambda x: 2*x + 3)
```

```mathematica
# 期待値の線形性
# a, b を実数, X を確率変数としたとき

E(aX + b) = aE(X) + b

# が成り立つ
```

```python
E(2X + 3) が 2E(X) + 3 と等しいか
2 * E(X) + 3
```

#### 分散

```mathematica
V(X) = Σ(xk - μ)2 f(xk)
       k

# μ は確率変数 X の期待値で E(X) です
```
記号には σ（シグマ）を使って *σ<sup>2</sup>* や *V(X)* という表記がよく使われます.
なお, ただの *σ* は確率変数 *X* の標準偏差を表します

```python
mean = E(X)
np.sum([(x_k - mean) ** 2 * f(x_k) for x_k in x_set])
```

```python
def V(X, g=lambda x: x):
    x_set, f = X
    mean = E(X, g)
    return np.sum([(g(x_k) - mean) ** 2 * f(x_k) for x_k in x_set])
```

```python
V(X)

# Y = 2X + 3 の分散
V(X, lambda x: 2 * x + 3)
```

```mathematica
# 分散の公式
# a, b を実数, X を確率変数として

V(aX + b) = a2V(X)

# が成り立つ
```

```python
2 ** 2 * V(X)
```

## 5.2 ２次元の離散型確率変数

### 5.2.1 ２次元の離散型確率変数の定義

#### 同時確率分布
２次元の確率変数では, １次元の確率変数を２つ同時に扱い *(x, Y)* と表記します

```mathematica
{(xi, yi) | i = 1,2,...; j = 1,2,...}

P(X=xi, Y=yi) = Pij (i = 1,2,...; j=1,2,...)
```

このように確率変数 *(X,Y)* の振る舞いを同時に考えた分布のことを **同時確率分布（joint probability distribution）** または単に同時分布といいます

２次元確率分布の確率は *x* と *y* を引数にとる関数とみることができます.
そのような *P(X =x, Y = y) = fXY(x, y)* となる関数 *fXY(x, y)* を**同時確率関数（joint probability function）** といいます.

#### 確率の性質
確率は0以上で全確率が１でなければなりません







