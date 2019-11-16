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
```
