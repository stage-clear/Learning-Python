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

#### 確率の性質

> 確率は絶対に0以上で, すべての確率を足すと１にならなければなりません

