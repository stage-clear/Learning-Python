# 2. NumPy の基礎

```python
import numpy
numpy.__version__
```

## 2.1 Python のデータ型について
Pythonでは型が動的に推測されます

### 2.1.1 単なる整数ではないPythonの整数

### 2.1.2 単なるリストではないPythonのリスト

```python
L = list(range(10))
L
type(L[0])
```

```python
L2 = [str(c) for c in L]
L2
type(L2[0])
```

```python
L3 = [True, '2', 3.0, 4]
[type(item) for item in L3]
```

### 2.1.3 Python の固定型配列

```python
import array
L = list(range(10))
A = array.array('i', L)
A
# i は配列の要素が整数であることを指定する型コードです
```

NumPy の ndarray はさらに高い利便性を提供します

### 2.1.4 Python のリストから作る配列

```python
import numpy as np

# Pythonリストからの配列作成
np.array([1,4,3,2])

# NumPy配列の要素はすべて同じ型という制約があります
# 以下では、不動小数点にアップキャストされます
np.array([3.14, 4, 2 ,3])

# 配列のデータ型を明示的に設定する場合は dtypeキーワードを使用します
np.array([1,2,3,4], dtype='float32')

# NumPy配列は多次元にすることも可能
np.array([range(i, i+ 3) for i in [2, 4, 6]])
```

### 2.1.5 配列の構築

```python
# 要素がすべて0である長さが10の整数配列を作る
np.zeros(10, dtype=int)

# 要素がすべて1である3行5列の不動小数点数配列を作る
np.ones((3, 5))

# 要素がすべて3.14である3行5列の配列を作る
np.full((3, 5), 3.14)

# 開始値0, 終了値20で2ずつ増加する線形シーケンス配列を作る
np.arange(0, 20, 2)

# 0と1の間に均等に配置された5つの値の配列を作る
np.linspace(0, 1, 5)

# 0と1の間に均一に分布したランダムな値の3行3列の配列を作る
np.random.random((3, 3))

# 平均0と標準偏差1の正規分布乱数で3行3桁の配列を作る
np.random.normal(0, 1, (3, 3))

# 区間[0, 10) （0以上10未満）のランダムな整数で3行3桁の配列を作る
np.random.randint(0, 10, (3, 3))

# 3行3列の単位行列を作る
np.eye(3)

# 3つの整数の初期化されていない配列を作る
# 各要素の値は, そのメモリ位置にすでに存在していたものになる
np.empty(3)
```

### 2.1.6 NumPy の標準データ型

```python
# 文字列を使用して指定できます
np.zeros(10, dtype='int16')

# または、関連するNumPyオブジェクトを使用します
np.zeros(10, dtype=np.int16)
```

<img src="numpy_dtype.png" width="600" height="auto">

## 2.2 NumPy 配列の基礎

### 2.2.1 NumPy 配列の属性

```python
import numpy as np

# 同じ乱数を得るために, 乱数シードを設定する
np.random.seed(0)

x1 = np.random.randint(10, size=6) # 1次元配列
x2 = np.random.randint(10, size=(3, 4)) # 2次元配列
x3 = np.random.randint(10, size=(3, 4, 5)) # 3次元配列

# 各配列には, 属性として ndim（次元数), shape（各次元のサイズ）, size（配列の合計サイズ）を持ちます
print('x3 ndim:', x3.ndim)
print('x3 shape:', x3.shape)
print('x3 size:', x3.size)
print('x3 dtype:', x3.dtype)
print('x3 itemsize:', x3.itemsize)
print('x3 nbytes:', x3.nbytes)
```

### 2.2.2 配列インデクス: 配列の要素にアクセスする

```python
x1
x1[0]
x1[4]
x1[-1]
x1[-2]

x2
x2[0, 0]
x2[2, 0]
x2[2, -1]

# インデクスを使って値を変更できます
x2[0, 0] = 12

# 配列は固定型
# 以下では, x1はint型なので小数点は切り捨てられます
x1[0] = 3.14
```

### 2.2.3 配列のスライス: 部分配列にアクセスする

```python
x[start:stop:step]
```

### 2.2.3.1 1次元配列のスライス

```python
x = np.arange(10)

# 最初の５要素
x[:5]

# インデクス5以降の要素
x[5:]

# 中間の部分配列
x[4:7]

# 1つおきの要素
x[::2]

# インデクス1から始まる1つおきの要素
x[1::2]

# 逆順に全ての要素
x[::-1]

# インデクス5から逆順に1つおきの要素
x[5::-2]
```

#### 2.2.3.2 多次元配列のスライス

```python
x2

# 2行と3列
x2[:2, :3]

# すべての行と, 1つおきの列
x2[:3, ::2]

# x2の最初の列
print(x2[:, 0])

# x2の最初の行
print(x2[0, :])
```
