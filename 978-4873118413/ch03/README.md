# 3. pandasを使ったデータ操作
<sup>118~238</sup>

## 3.1 pandas のインストールと使用方法

```python
import pandas
pands.__version__
```

## 3.2 pandas オブジェクトの基礎
pandasオブジェクトとは行と列が単純な整数インデクスではなくラベルで識別できるNumPy構造化配列の拡張版と考えることができます

```python
import numpy as np
import pandas as pd
```

### 3.2.1 Seriesオブジェクト

```python
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data

# values は, おなじみのNumPy配列です
data.values

data.index

data[1]

data[1:3]
```

#### 3.2.1.1 一般化NumPy配列としてのSeries
- SeriesとNumPy配列の本質的な違いはインデクス
  - NumPy配列は暗黙的に定義された整数インデクス
  - Seriesには値に関連付けられた明示的に定義されたインデクス

```python
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data

data['b']

# インデクスは連続していなくても, 順番に並んでいなくても構いません
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[2, 5, 3, 7])
data

data[5]
```

#### 3.2.1.2 特殊辞書としてのSeries
Python辞書から直接Seriesを構築することで, Seriesが辞書の１種であるという比喩をもっと明確にできます.

```python
population_dict = {
  'California': 38332521,
  'Texas': 26448193,
  'New York': 19651127,
  'Florida': 19552860,
  'Illinois': 12882135
}

population = pd.Series(population_dict)
population

population['California']

population['California':'Illinois']
```

#### 3.2.1.3 Seriesオブジェクトの作成

```python
pd.Series([2, 4, 6])

pd.Series(5, index=[100, 200, 300])

pd.Series({2: 'a', 1: 'b', 3: 'c'})

pd.Series({2: 'a', 1: 'b', 3: 'c'}, index=[3,2])
```

### 3.2.2 DataFrameオブジェクト

#### 3.2.2.1 一般化NumPy配列としてのDataFrame
DataFrame は, 柔軟な行インデクスと柔軟な列名の両方を持つ２次元配列に例えられます

```python
area_dict = {
    'California': 423967,
    'Texas': 695662,
    'New York': 141297,
    'Florida': 170372,
    'Illinois': 149995
}
area = pd.Series(area_dict)
area

states = pd.DataFrame({
    'population': population,
    'area': area
})
states

states.index
states.columns
```

#### 3.2.2.2 特殊辞書としてのDataFrame

```python
states['area']
```

#### 3.2.2.3 DataFrameオブジェクトの作成
**Seriesオブジェクトから作成する**

```python
pd.DataFrame(population, columns=['population'])
```

**辞書のリストから作成する**

```python
data = [{'a': i, 'b': 2 * i} for i in range(3)]
pd.DataFrame(data)

# 辞書内にいくつか値が見つからなくても, pandasはそれらをNaN値で埋めます
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
```

**Seriesオブジェクトの辞書から作成する**

```python
pd.DataFrame({
    'population': population,
    'area': area
})
```

**２次元NumPy配列から作成する**

```python
pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c'])
```

**NumPy構造化配列から作成する**

```python
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
pd.DataFrame(A)
```

### 3.2.3 Indexオブジェクト

```python
ind = pd.Index([2,3,5,7,11])
ind
```

#### 3.2.2.1 不変配列としてのIndex

```python
ind[1]

ind[::2]

print(ind.size, ind.shape, ind.ndim, ind.dtype)
```

IndexオブジェクトとNumPy配列の違いの１つは, インデクスが不変であることです

```python
ind[1] = 0
```

#### 3.2.3.2 順序付き集合（set）としてのIndex
- 和集合（union）
- 積集合（intersection）
- 差集合（difference）

```python
indA = pd.Index([1,3,5,7,9])
indB = pd.Index([2,3,5,7,11])

indA & indB # 積集合（どちらにも含まれる）

indA | indB # 和集合（すべて）

indA ^ indB # 差集合（片方だけに含まれる）
```

## 3.3 インデクスとデータの選択

### 3.3.1 Seriesのデータ選択
#### 3.3.1.1 辞書としてのSeries

```python
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
data

data['b']

'a' in data

data.keys()

list(data.items())

data['e'] = 1.25
```

#### 3.3.1.2 １次元配列としてのSeries

```python
# 明示的なインデクスによるスライス
data['a':'c']

# 間接的な整数インデクスによるスライス
data[0:2]

# マスク
data[(data > 0.3) & (data < 0.8)]

# ファンシーインデクス
data[['a', 'b']]
```

#### 3.3.1.3 インデクス属性: loc, iloc: ix

```python
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data

# 明示的なインデクス指定
data[1]

# スライスの場合の, 間接的な指定
data[1:3]
```

```python
# loc属性
# 明示的なインデクスを使ったインデクスおよびスライス
data.loc[1]
data.loc[1:3]

# iloc属性
# 間接的なPythonスタイルのインデクスを使ったインデクスとスライス
data.iloc[1]
data.iloc[1:3]
```

### 3.3.2 DataFrameのデータ選択
#### 3.3.2.1 辞書としてのDataFrame

```python
area = pd.Series({
    'California': 42369,
    'Texas': 695543,
    'New York': 141295,
    'Florida': 170327,
    'Illinois': 149953
})
pop = pd.Series({
    'California': 38335422,
    'Texas': 2644987,
    'New York': 1965789,
    'Florida': 1955280,
    'Illinois': 12880934
})

data = pd.DataFrame({
    'area': area,
    'pop': pop
})
data

data['area']
data.area
data.area is data['area']

data['density'] = data['pop'] / data['area']
```

#### 3.3.2.2 ２次元配列としてのDataFrame

```python
data.values

# DataFrameを転置して行
と列を交換できます
data.T
```

```python
# 配列形式に対してインデクスを１つ指定すると, 行が返る
data.values[0]

# DataFrameにインデクスを１つ指定すると列が返る
data['area']
```

DataFrameに配列形式のインデクスを使うには, loc, iloc, ix を使う

```python
data.iloc[:3, :2]

data.loc[:'Illinois', :'pop']

data.ix[:3, :'pop']
```

* ixはdeprecated

```python
# locを使って, マスクとファンシーインデクスを組み合わせる
data.loc[data.density > 100, ['pop', 'density']]

data.iloc[0, 2] = 90 # 1行3列
data
```

#### 3.3.2.3 その他のインデクス規則
**インデクス**は列を参照しますが, **スライス**は行を参照します

```python
# スライスは行を参照します
data['Florida':'Illinois']
data[1:3]

# マスキング操作も行に対して解釈されます
data[data.density > 50]
```

## 3.4 pandasデータの操作

### 3.4.1 ufunc: インデクスの保存
```python
import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser


df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])
df

# これらのオブジェクトにNumPy ufuncを適用すると, 結果はインデクスが保存された別のpandasオブジェクトになります

np.exp(ser)
np.sin(df * np.pi / 4)
```

### 3.4.2 ufunc: インデクスの整列
２つの異なるデータソースを結合し, 米国の面積上位3つの州と 人口上位3つの州を検索する

```python
area = pd.Series({
    'Alaska':  172337,
    'Texas': 695662,
    'California': 423697
}, name='area')

population = pd.Series({
    'California': 38332521,
    'Texas': 26448193,
    'New York': 19651127
}, name='population')

population / area
# 結果の配列は, 和集合で構成されます

area.index | population.index
```

```python
A = pd.Series([2,4,6], index=[0,1,2])
B = pd.Series([1,3,5], index=[1,2,3])
A + B

# 欠損値を埋める値を明示的に指定することができます
A.add(B, fill_value=0)
```

#### 3.4.2.2 DataFrame オブジェクトのインデクス整列

```python
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list('AB'))
B = pd.DataFrame(rng.randint(0, 10, (3, 3)), columns=list('BAC'))
A + B # この結果には欠損値がある

fill = A.stack().mean()
A.add(B, fill_value=fill)
```

### 3.4.3 ufunc: DataFrameとSeriesの演算
DataFrameとSeries間で操作を実行する場合, インデクスと列の配置も同時に維持されます

```python
# 2次元配列と, その中の1行との差を計算します
A = rng.randint(10, size=(3, 4))
A

A - A[0]
```

NumPy のブロードキャストルールによると, 2次元配列とその中の1行との減算は, 行単位で行われます.

```python
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]
# pandas でもデフォルトでは行単位で計算が行われます

# 列単位で操作する場合は, axisキーワードを指定することで可能になります
df.subtract(df['R'], axis=0)

# DataFrame と Series との操作は, 2つの要素間のインデクスが自動的に揃えられます
halfrow = df.iloc[0, ::2]
halfrow

df - halfrow
```

## 3.5 欠損値の扱い
欠落しているデータを一般的に **null値**, **NaN値**, または **NA値**と呼びます.

### 3.5.1 欠損値表現のトレードオフ

### 3.5.2 pandas の欠損値

### 3.5.2.1 None: Pythonの欠損値

```python
import numpy as np
import pandas as pd

vals1 = np.array([1, None, 3, 4])
vals1
```

#### 3.5.2.2 NaN: 数値データの欠損値

```python
vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype

# NaNはデータに対する一種のウイルスのようなもので, 他のあらゆるオブジェクトに感染します.
# つまり, 演算に関係なくNaNをしようした　算術演算の結果は, NaNになります
vals2.sum(), vals2.min(), vals2.max()

# NumPy はこれらの欠損値を無視する特別な集約手段を提供しています
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)
```

#### 3.5.2.3 pandas における NaN と None

```python
pd.Series([1, np.nan, 2, None])

x = pd.Series(range(2), dtype=int)
x

x[0] = None
x
# None は自動的にNaNに変換saremasu
```

### 3.5.3 null値が存在する場合の処理

null値を検出, 削除, および置換するための有用なメソッドがいくつかあります

- isnull() - 欠損値の存在を示すブール値マスク配列を作成する
- notnull() - isnull() の逆
- dropna() - データから欠損値を取り除いたデーターを作る
- fillna() - 不足している要素に値を埋め込んだデータのコピーを返す

#### 3.5.3.1 null値の検出
isnull(), notnull()

```python
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.notnull()]
```

#### 3.5.3.2 欠損値の除外
dropna(), fillna()

```python
data.dropna()
```

```python
df = pd.DataFrame([
  [1, np.nan, 2],
  [2, 3, 5],
  [np.nan, 4, 6]
])

df.dropna()
# dropna() は null値が存在するすべての行を削除します

df.dropna(axis='columns')
```

```python
df[3] = np.nan
df

# how='all' を指定することで, すべての要素がnull値である行/列のみが削除されます
df.dropna(axis='columns', how='all')

# threshパラメータを使用して, 行または列を維持するためのnull以外の値の最小個数を指定できます
# (null値以外の要素がN個以上ある)
df.dropna(axis='rows', thresh=3)
```

#### 3.5.3.3 欠損値への値設定

```python
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data

# 欠損値を0で埋める
data.fillna(0)

# 欠損値を, 欠損値の一つ前の値を埋める（forward fill）
data.fillna(method='ffill')

# 欠損値を, 欠損値の一つ後ろの値を埋める（back fill）
data.fillna(method='bfill')
```

```python
df

df.fillna(method='ffill', axis=1)
```

forward fill を行う際に前の値が利用できない場合, NA値が残る点に注意が必要です

## 3.6 階層型インデクス

```python
import pandas as pd
import numpy as np
```

### 3.6.1 多重インデクスを持つSeries
1次元のSeriesで2次元データを表現する方法

#### 3.6.1.1 誤った手法

```python
index = [
    ('California', 2000),
    ('California', 2010),
    ('New York', 2000),
    ('New York', 2010),
    ('Texas', 2000),
    ('Texas', 2010)
]
populations = [
    33871648,
    37253956,
    18976457,
    19378102,
    20185182,
    25145561,
]
pop = pd.Series(populations, index=index)

pop[('California', 2010):('Texas', 2000)]
```

```python
pop[[i for i in pop.index if i[1] == 2010]]
```

#### 3.6.1.2 より良い手法: MultiIndex

```python
index = pd.MultiIndex.from_tuples(index)
index
```

```python
pop = pop.reindex(index)
pop
```

```
# この結果は, 指定したキーを１つだけ持つインデクス付き配列になります
pop[:, 2010]
```

#### 3.6.1.3 多次元に対する MultiIndex
unstack() メソッドは, 多重インデクスを使ったSeriesを, 一般的なインデクス付きDataFrameに変換します.

```python
pop_df = pop.unstack()
pop_df
```

もちろん stack() メソッドは逆の操作を行います.

```python
pop_df.stack()
```

多重インデクスを使えば１次元のSeries内で２次元を表せるように, SeriesやDataFrameで３次元以上のデータを表せるからです

```python
pop_df = pd.DataFrame({
    'total': pop,
    'under18': [
        9267089, 9284094,
        4689743, 4318033,
        5906301, 6879014
    ]
})
```

### 3.6.2 MultiIndex の作成方法

```python
df = pd.DataFrame(
    np.random.rand(4, 2),
    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
    columns=['data1', 'data2']
)
```

### 3.6.2.1 明示的なMultiIndexの作成
pd.MultiIndex

```python
# 配列のリストから MultiIndex を作成
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])

# タプルのリストから MultiIndex を作成
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])

# それぞれのインデクスのデカルト積から MultiIndex を作成
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
```

#### 3.6.2.2 MultiIndexのレベル名
names引数を渡すか, sakuseigoniインデクスのnames属性を設定する

```python
pop.index.names = ['state', 'year']
```

#### 3.6.2.3 列に対するMultiIndex

```python
# 階層化した列とインデクス
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], names=['subject', 'type'])

# データは適当に埋める
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

# DataFrame の作成
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
```

### 3.6.3 MultiIndex のインデクス指定とスライス
#### 3.6.3.1 多重インデクスSeries

```python
pop['California', 2000]

pop['California']

pop[:, 2000]

pop[pop > 2200000]

pop[['California', 'Texas']]
```

#### 3.6.2.2 多重インデクスDataFrame
DataFrameでは列が優先され, 多重インデクス付きSeriesで使用した構文は列に適用されることに注意してください

```python
health_data

health_data['Guido', 'HR']

# loc, iloc を使用することもできます
health_data.iloc[:2, :2]

# loc, iloc はタプルを使って複数のインデクスを指定できます
health_data.loc[:, ('Bob', 'HR')]

# タプル内のインデクスにはスライスを指定できません
health_data.loc[(:, 1), (:, 'HR')] # 構文エラー

# 
idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]
```

### 3.6.4 多重インデクスの並べ替え

```python
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']

try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)
```

```python
data = data.sort_index()
data
```

```python
data['a':'b']
```

#### 3.6.4.2 インデクスのstackのunstack
２次元に変換する場合, 必要に応じてレベルを指定することもできます

```python
pop.unstack(level=0)

pop.unstack(level=1)

# unstack()の逆操作stack()を用いて元のSeriesを復元できます
pop.unstack().stack()
```

#### 3.6.4.3 インデクスの設定と再設定

```python
pop_flat = pop.reset_index(name='population')
pop_flat
```

```python
pop_flat.set_index(['state', 'year'])
```

### 3.6.5 多重インデクスに基づいたデータ集約
mean(), sum(), max() など

```python
health_data

# レベルを指定した平均
data_mean = health_data.mean(level='year')
data_mean

# axis キーワードを使用すれば, 列レベル間での平均が得られます
data_mean.mean(axis=1, level='type')
```

## 3.7 データセットの連結: concatとappend

```python
import pandas as pd
import numpy as np
```

```python
def make_df (cols, ind):
    """DataFrameの簡易作成関数"""
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

# サンプル DataFrame
make_df('ABC', range(3))
```

### 3.7.1 再掲: NumPy配列の連結
np.concatenate()

```python
x = [1,2,3]
y = [4,5,6]
z = [7,8,9]

np.concatenate([x, y, z])
```

```python
x = [[1,2], [3,4]]
np.concatenate([x, x], axis=1)
```

### 3.7.2 pd.concatを使った単純な連結
pd.concat()

```python
# pandas v0.18 の関数シグニチャ
pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
    keys=None, levels=None, names=None, verify_integrity=False,
    copy=True)
```

```python
ser1 = pd.Series(['A', 'B', 'C'], index=[1,2,3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4,5,6])
pd.concat([ser1, ser2])
```

```python
# DataFrameによる高次元のオブジェクトを連結
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(df1)
print(df2)
print(pd.concat([df1, df2]))
```

```python
# 連結が行われる軸を指定
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(df3)
print(df4)
print(pd.concat([df3, df4], axis=1))
```

#### 3.7.2.1 インデクスの重複
重複するインデクスを持っていても, pd.concat ではインデクスが保持される

```python
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index # インデクスを重複させる
print(x)
print(y)
print(pd.concat([x, y]))
```

##### 重複をエラーとして補足する
verify_integrity フラグを指定する

```python
try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print('ValueError:', e)
```

##### インデクスを無視する
ignore_index フラグを設定する

```python
print(x)
print(y)
print(pd.concat([x, y], ignore_index=True))
# 新しい整数インデクスが作成されます
```

##### MultiIndexキーを追加する
keysオプションを使用してデータソースのラベルを指定する

```python
print(x)
print(y)
print(pd.concat([x, y], keys=['x', 'y']))
```

#### 3.7.2.2 joinによる連結

```python
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
print(df5); print(df6); print(pd.concat([df5, df6]))
```

和集合（join='outer'）, 積集合（join='inner'）
```python
print(df5); print(df6);
print(pd.concat([df5, df6], join='inner'))
```

```python
print(df5); print(df6);
print(pd.concat([df5, df6], join_axes=[df5.columns]))
```
* join-axes は deprecated

#### 3.7.2.3 appendメソッド

```python
print(df1); print(df2)
print(df1.append(df2))
```

Pythonリストのappend()とextend()メソッドとは異なり, pandasのappend()メソッドは元のオブジェクトを変更せず, 結合されたデータで新しいオブジェクトを作成することに注意してください

## 3.8 データセットの結合: mergeとjoin
pandasが提供する重要な機能の１つは, 高パフォーマンスなメモリ内joinおよびmerge操作です

### 3.8.1 関係代数

### 3.8.2 結合の種類

#### 3.8.2.1 1対１結合

```python
import pandas as pd
import numpy as np
```

```python
df1 = pd.DataFrame({
    'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']
})
df2 = pd.DataFrame({
    'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
    'hire_date': [2004, 2008, 2012, 2014]
})

print(df1)
print(df2)
```

```
df3 = pd.merge(df1, df2)
df3
```

#### 3.8.2.2 多対1結合

```python
df4 = pd.DataFrame({
    'group': ['Accounting', 'Engineering', 'HR'],
    'supervisor': ['Carly', 'Guido', 'Steve']
})

print(df3); print(df4);
print(pd.merge(df3, df4))
```

#### 3.8.2.3 多対多結合

```python
df5 = pd.DataFrame({
    'group': ['Accounting', 'Accounting', 'Engineering', 'Engineering', 'HR', 'HR'],
    'skills': ['math', 'spreadsheets', 'coding', 'linux', 'spreadsheets', 'organization']
})

print(df1); print(df5);
print(pd.merge(df1, df5))
```

### 3.8.3 キーの指定
列名が一致しないこともしばしば発生します。
pd.merge() はこれを扱うためのさまざまなオプションを提供します

#### 3.8.3.1 onキーワード
列名または列名リストを与えてキーを明示的に指定する

```python
print(pd.merge(df1, df2, on='employee'))
```

#### left_on キーワードと right_on キーワード

```python
df3 = pd.DataFrame({
    'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
    'salary': [70000, 80000, 120000, 90000]
})
print(df1); print(df3)
print(pd.merge(df1, df3, left_on='employee', right_on='name'))
# 左の employee と右の name でデータを一致させる
```

```python
pd.merge(df1, df3, left_on='employee', right_on='name').drop('name', axis=1)
# drop() メソッドで列を削除できます
```

### 3.8.3.3 left_index キーワードと right_index キーワード
列を結合するのではなく, インデクスを結合する

```python
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(df1a); print(df2a)
```

```python
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
```

利便性のために, DataFrameはデフォルトでインデクスをキーとして結合を実行する join() メソッドを提供しています
```python
print(df1a.join(df2a))
```

left_index と right_on または left_on と right_index を組み合わせれば, インデクスと列の混在した指定が可能です

```python
print(pd.merge(df1a, df3, left_index=True, right_on='name'))
```

### 3.8.4 結合に対する集合演算の指定
