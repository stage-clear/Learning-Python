# DataFrame の基礎

```python
import pandas
```

## 1.2 最初のデータセットをロードする
```python
df = pandas.read_csv('../data/gapminder.tsv', sep='\t')
print(df.head())
```

```python
print(type(df))
```

```python
# 行と列の個数をとる
print(df.shape)

# 列の名前を見る
print(df.columns)

# 各列のdtypeを見る
print(df.dtypes)

# データの詳しい情報を見る
print(df.info())
```

## 1.3 列、行、セルを見る

### 1.3.1 列を絞り込む
#### 1.3.1.1 名前で列を絞り込む

```python
# country の列だけを取り出して独自の変数に保存する
country_df = df['country']

# 最初の５この値を表示する
print(country_df.head())

# 最後の５個を表示する
print(country_df.tail())

# country と continent と year のデータを見る
subset = df[['country', 'continent', 'year']]

print(subset.head())

print(subset.tail())
```

### 1.3.2 行を絞り込む
#### 1.3.2.1 インデックスラベルによる行の抽出
```python
print(df.head())

# 最初の行を抽出する
# (Pythonでは0から数える)
print(df.loc[0])

# 100番目の行を抽出する
print(df.loc[99])

# 最後の行を抽出する （これはエラーになる）
print(df.loc[-1])
# 最後の行を抽出する （正解）
number_of_rows = df.shape[0]
last_row_index = number_of_rows - 1
print(df.loc[last_row_index])

# 望みの結果を得る方法は数多く存在する
print(df.tail(n=1))
```

```python
# tail と loc の違い
subset_loc = df.loc[0]
subset_head = df.head(n=1)

# loc を使って得られる１行の型
print(type(subset_loc))
# -> <class 'padas.core.series.Series'>

# headを使って得られる１行の型
print(type(subset_head))
# -> <class 'pandas.core.frame.DataFrame'>
```

**複数行の抽出**

```python
# 第1行と第100行と第1000行を選択する
# 複数列の選択に使ったのと同じ
# ２重角カッコの構文に注目
print(df.loc[[9, 99, 999]])
```

#### 1.3.2.2 インデックス番号による行の抽出

```python
# 2番目の行を見る
print(df.iloc[1])

# 100番目の行を見る
print(df.iloc[99])

# -1を使って最後の行を抽出する
print(df.iloc[-1])

# 第1行と第100行と第1000行を選択する
print(df.iloc[[0, 99, 999]])
```

#### 1.3.2.3 ix による行の抽出（pandas v0.20以前）

### 1.3.3 組み合わせて絞り込む
#### 1.3.3.1 複数行の抽出
Python のスライス構文は `:` を使う。コロンだけを指定すると、その属性は「すべて」を意味する。
`df.loc[:, [columns]]`

```python
# loc で列を絞り込む
# コロンの位置に注意しよう
# これは、全部の行を選択する
subset = df.loc[:, ['year', 'pop']]
print(subset.head())

# iloc で列を絞り込む
# iloc では整数値を使える
# -1 は最後の列を選択する
subset = df.iloc[:, [2, 4, -1]]
print(subset.head())

# loc で列を絞り込む
# ただし、整数値を指定しているのでエラーになる
subset = df.loc[:, [2, 4, -1]]
print(subset.head())

# iloc で列を絞り込む
# ただしインデックス名を指定しているのでエラーになる
subset = df.iloc[:, ['year', 'pop']]
print(subset.head())
```

#### 1.3.3.2 範囲による複数列の抽出
```python
# 0から4までの整数を含む範囲を作成
small_range = list(range(5))
print(small_range)
# > [0,1,2,3,4]

# その範囲で DataFrame オブジェクトを絞り込む
subset = df.iloc[:, small_range]
print(subset.head())

# 3から5までの整数を含む範囲を作成
small_range = list(range(3, 6))
print(subset.head())
# > [3,4,5]
subset = df.iloc[:, small_range]
print(subset.head())

# 0 から 5に至る、1つおきの整数を含む範囲を作成
small_range = list(range(0, 6, 2))
subset = df.iloc[:, small_range]
print(subset.head())
```

#### 1.3.3.3 整数列のスライシング

```python
small_range = list(range(3))
subset = df.iloc[:, small_range]
print(subset.head())

# 最初の３つの列をスライスする
subset = df.iloc[:, :3]
print(subset.head())

small_range = list(range(3, 6))
subset = df.iloc[:, small_range]
print(subset.head())

# 3から5までを含む列をスライス
subset = df.iloc[:, 3:6]
```

#### 1.3.3.4 列と行の抽出
```python
#locを使うとき
print(df.loc[42, 'country'])

# ilocを使うとき
print(df.iloc[42, 0])

# これはエラーになるだろう
print(df.loc[42, 0])

# データから43番目のcountryを取り出す
print(df.loc[42, 'country'])

# 'country'の代わりにインデックス0を使う (非推奨)
df.ix[42, 0]
```

#### 1.3.3.5 複数行、複数列の抽出
```python
# 第1行、第100行、第1000行を、
# 第1列、第4列、第６列から切り出す
print(df.iloc[[0, 99, 999], [0, 3, 5]])

# インデックスではなく、列名を直接使えば
# もっとコードが読みやすくなる。ただし、
# それには、ilocではなくlocを使う必要がある
print(df.[[0,99,999], ['country', 'lifeExp', 'gdpPercap']])

# locおよびilocの属性では、行の位置指定にスライス構文を使える
print(df.loc[10:13, ['country', 'lifeExp', 'gdpPercap']])
```

## 1.4 グループ化と集約
### 1.4.1 グループごとの平均値
```python
# 平均値(mean)
# 処理のプロセス「分割-適用-結合」（split-apply-combine）
print(df.groupby('year')['lifeExp'].mean())

grouped_year_df = df.groupby('year')
grouped_year_df_lifeExp = grouped_year_df['lifeExp']
mean_lifeExp_by_year = grouped_year_df_lifeExp.mean()
print(mean_lifeExp_by_year)

# バックスラッシュを使えば、長い１行のコードを複数行に分けることができる
multi_group_var = df.\
  groupby(['year', 'continent'])\
  [['lifeExp', 'gdpPercap']].\
  mean()

multi_group_var = df.\
  groupby(['year', 'continent'])\
  [['lifeExp', 'gdpPercap']].\
  mean()
print(multi_group_var)

# この DataFrame オブジェクトを平坦化
flat = multi_group_var.reset_index()
print(flat.head())
```

### 1.4.2 グループごとの度数/頻度
```python
# nunique() : 重複を除くユニークな出現回数
# value_counts() : 重複を含む出現回数
print(df.groupby('contient')['country'].nunique())
```

## 1.5 基本的なグラフ
```python
import matplotlib.pyplt as plt

global_yearly_life_expectancy.plot()
plt.show()
```
