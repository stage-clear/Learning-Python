# 9. applyによる関数の適用
## 9.1 はじめに
**目標**
1. 関数
2. データの列または行に関数を適用する

## 9.2 関数

```python
def my_function():
    # ４個のスペースでインデントして
    # 関数のコードを書く
```

```python
def my_sq(x):
    """与えられた値を2乗する"""
    return x ** 2

def avg_2(x, y):
    """２つの数の平均値を計算する***
    return (x + y) / 2

print(my_sq(4))
print(avg_2(10, 20))
```
## 9.3 apply の基礎
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.apply.html

```python
import pandas as pd

df = pd.DataFrame({
    'a': [10, 20, 30],
    'b': [20, 30, 40]
})
print(df)
```

```python
# 単純な書き方でも、列を直接2乗することができる
print(df['a'] ** 2)
```

### 9.3.1 Series に適用する
```python
# 最初の列をとる
print(type(df['a']))

# 最初の行をとる
print(type(df.iloc[0]))

# 自作の関数を列aに適用
sq = df['a'].apply(my_sq)
print(sq)
```

```python
def my_exp (x, e):
    return x ** e

# 第２パラメータをキーワード引数として apply に渡す
ex = df['a'].apply(my_exp, e=2)
print(ex)

ex = df['a'].apply(my_exp, e=3)
print(ex)
```

### 9.3.2 DataFrame に適用する

```python
df = pd.DataFrame({
    'a': [10, 20, 30],
    'b': [20, 30, 40]
})

print(df)
```

> ある関数を１個のDataFrameに適用するには、関数を適用すべき軸(axis)を指定する必要がある<br>
> もし関数で列を扱いたければ、apply に axis=0 を渡す。行を扱いたければ axis=1 を渡す

```python
def print_me (x):
    print(x)
```

#### 9.3.2.1 列ごとの演算

```python
df.apply(print_me, axis=0)
```

```python
# この関数は列ごとに適用しようとするとエラーになる
# 列の全体が第１引数として渡されているため
def avg_3 (x, y, z):
    return (x + y + z) / 3
print(df.apply(avg_3))

# 次のように書き換えられる
def avg_3_apply(col):
    x = col[0]
    y = col[1]
    z = col[2]
    return (x + y + z) / 3

df.apply(avg_3_apply)
```

#### 9.3.2.2 行ごとの演算

```python
# エラーになる
print(df.apply(avg_3_apply, axis=1))

# 書き直す
def avg_2_apply (row):
    x = row[0]
    y = row[1]
    return (x + y) / 2

print(df.apply(avg_2_apply, axis=1))
```

## 9.4 apply の応用

```python
import seaborn as sns

titanic = sns.load_dataset('titanic')

print(titanic)
```

このデータに null または NaN の値がいくつあるかを計算し、各列または各行における完全なケースの比率を求める

```python
# 1. 欠損値を求める
# numpy の sum 関数を使う
import numpy = as np

def count_missing (vec):
    """ベクトルにある欠損値の数を数える"""
    # 値が欠損しているかどうかを示す
    # 真偽値のベクトルをとる
    null_vec = pd.isnull(vec)

    # null の値は sum に影響を与えない
    # null_vec に対する sum の計算で欠損値の数がわかる
    null_count = np.sum(null_vec)

    # ベクトルにある欠損値を返す
    return null_count

# 2. 欠損率を求める
def prop_missing (vec):
    """ベクトルで欠損値が閉める比率"""
    # 分子（numerator）は欠損値の数
    # 上で定義したcount_missing関数を使う!
    num = count_missing(vec)
    
    # 分母(denominator)はベクトルにある値の総数
    # これには欠損値も含まれる
    dem = vec.size
    
    # 欠損値の比率（proportion）を返す
    return num / dem

def prop_complete (vec):
    """ベクトルで非欠損値が占める比率"""
    # すでに書いた prop_missing 関数を利用し、
    # その値を1から差し引く
    return 1 - prop_missing(vec)
```

### 9.4.1 列ごとの演算
```python
cmis_col = titanic.apply(count_missing)
pmis_col = titanic.apply(prop_missing)
pcom_col = titanic.apply(prop_complete)

print(cmis_col)
print(pmis_col)
print(pcom_col)
```

embark_town の列には欠損値が２つしかない。その２つの行を見れば、これらの値がランダムに欠損したのか、それとも何かあって欠損したのかを調べることができるだろう。

```python
print(titanic.loc[pd.isnull(titanic.embark_town), :])
```

### 9.4.2 行ごとの演算
axis=1

```python
cmis_row = titanic.apply(count_missing, axis=1)
pmis_row = titanic.apply(prop_missing, axis=1)
pcom_row = titanic.apply(prop_complete, axis=1)

print(cmis_row.head())
print(pmis_row.head())
print(pcom_row.head())

# データの中に複数の欠損値を持つ行がどのくらいあるか
print(cmis_row.value_counts())

# これらの値を含む新しい列を作ることもできる
titanic['num_missing'] = titanic.apply(count_missing, axis=1)

print(titanic.head())

# 複数の欠損値を持つ行を見ることもできる
print(titanic.loc[titanic.num_missing > 1, :].sample(10))
```

## 9.5 関数のベクトル化
vectorize関数とでコレータを使うと、どんな関数でもベクトル化できる

```python
df = pd.DataFrame({
    'a': [10, 20, 30],
    'b': [20, 30, 40]
})

print(df)
```

```pytyon
def avg_2 (x, y):
    return (x + y) / 2
```

> `avg2(df['a'], df['b'])` と書けるようにし、それぞれの結果として `[15, 25, 35]` を得られるようにしたい

```python
print(avg_2(df['a'], df['b']))
# このアプローチが使えるのは、関数内部で行われる計算の性質が、もともとベクトル化に適しているからだ


import numpy as np

def avg_2_mod (x, y):
    if (x == 20):
        return (np.NaN)
    else:
        return (x + y) / 2

# エラーになる
print(avg_2_mod(df['a'], df['b']))

# 期待通りに動作する
print(avg_2_mod(10, 20))
print(avg_2_mod(20, 30))
```

### 9.5.1 Numpy を使ったベクトル化
https://docs.python.org/ja/3/reference/compound_stmts.html#function-definitions

```python
# np.vectorize により実際には新しい関数を作る
avg_2_mod_vec = np.vectorize(avg_2_mod)
print(avg_2_mod_vec(df['a'], df['b']))
```

Pythonのデコレータ(decorator)を使うことで、新しい関数を作成せずに、既存の関数を自動的にベクトル化することが可能だ

```python
# でコレータを使ってベクトル化するには、
# 関数定義の前に@記号を使う
@np.vectorize
def v_avg_2_mod (x, y):
    """xが20でなければ平均値を計算する。前と同様だが、デコレータでベクトル化する"""

    if (x == 20):
        return np.NaN
    else:
        return (x + y) / 2

# 上記のように書くことで、新しい関数を作ることなく
# ベクトル化した関数を直接使える
print(v_avg_2_mod(df['a'], df['b']))
```

### 9.5.2 numba を使ったベクトル化

```python
import numba

@numba.vectorize
def v_avg_2_numba (x, y):
    # 関数に型情報を追加する必要がある
    if (int(x) == 20):
        return np.NaN
    else:
        return (x + y) / 2

# numbaはpandasのオブジェクトを認識できない
print(v_avg_2_numba(df['a'], df['b']))

# このため NumPy 配列表現を渡す必要がある
print(v_avg_2_numba(df['a'].values, df['b'].values))
```

## 9.6 ラムダ関数
applyメソッドの中で使う関数が十分にシンプルな時は、別の関数を作る必要がないかもしれない

```python
docs = pd.read_csv('data/doctors.csv', header=None)
```

```python
import regex

p = regex.compile('\w+\s+\w+')

def get_name (s):
    return p.match(s).group()

docs['name_func'] = docs[0].apply(get_name)

print(docs)

# ラムダで書き直す
docs['name_lamb'] = docs[0].apply(lambda x: p.match(x).group())
print(docs)
```

## 9.7 まとめ
