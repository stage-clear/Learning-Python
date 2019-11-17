# 4. データを組み立てる
##  4.2 整然データ
- 各行が１回の観測である
- 各行が１個の変数である
- 観察単位の型ごとに、１個の表が構成されている

### 4.2.1 データセットを組み合わせる

## 4.3 連結
連結を実現するのは、pandas の `concat` を使う

### 4.3.1 行の追加
```python
import pandas as pd

df1 = pd.read_csv('data/concat_1.csv')
df2 = pd.read_csv('data/concat_2.csv')
df3 = pd.read_csv('data/concat_3.csv')

print(df1)
print(df2)
print(df3)
```

```python
row_concat = pd.concat([df1, df2, df])
print(row_concat)

# 連結したDataFrameから第４行を抽出
print(row_concat.iloc[3,])
```

```python
# 新しいデータを作成
new_row_series = pd.Series(['n1', 'n2', 'n3', 'n4'])
print(new_row_series)

# 新しい行をDataFrameに追加すると..（これはうまくいかない）
print(pd.concat([df1, new_row_series]))

new_row_df = pd.DataFrame([['n1', 'n2', 'n3', 'n4']], columns=['A', 'B', 'C', 'D'])
print(new_row_df)
print(pd.concat([df1, new_row_df]))

# もし１個のオブジェクトを既存のDataFrameに追加したいだけなら append() で用が足りる
# DataFrame を append
print(df1.append(df2))
# 1行だけのDataFrame を append
print(df1.append(new_row_df))

# Pythonの辞書を append:
data_dict = {
  'A': 'n1',
  'B': 'n2',
  'C': 'n3',
  'D': 'n4'
}
print(df1.append(data_dict, ignore_index=True))
```

#### 4.3.1.1 インデックスの再設定
```python
row_concat_i = pd.concat([df1, df2, df3], ignore_index=True)
print(row_concat_i)
```

### 4.3.2 列の追加
concat関数の axis パラメータに1を設定する

```python
col_concat = pd.concat([df1, df2, df3], axis=1)
print(col_concat)

print(col_concat['A'])
```

```python
# DataFrameに1列追加するだけなら直接行うことができる
col_concat['new_col_list'] = ['n1', 'n2', 'n3', 'n4']
print(col_concat)

col_concat['new_col_series'] = pd.Series(['n1', 'n2', 'n3', 'n4'])
print(col_concat)
```

```python
# 列のインデックスもリセット可能であり、それで列名の重複を避けられる
print(pd.concat([df1, df2, df3], axis=1, ignore_index=True))
```

### 4.3.3. インデックスが異なる連結
#### 4.3.3.1 列が異なる行を連結する

```python
df1.columns = ['A', 'B', 'C', 'D']
df2.columns = ['E', 'F', 'G', 'H']
df3.columns = ['A', 'C', 'F', 'H']

print(df1)
print(df2)
print(df3)

row_concat = pd.concat([df1, df2, df3])
print(row_concat)

print(pd.concat([df1, df2, df3], join='inner'))

# 共通する列だけを返す
print(pd.concat([df1, df3], ignore_index=False, join='inner'))
```

#### 4.3.3.2 行が異なる列を連結する
```python
df1.index = [0,1,2,3]
df2.index = [4,5,6,7]
df3.index = [0,2,5,7]

print(df1)
print(df2)
print(df3)
```

```python
col_concat = pd.concat([df1, df2, df3],, axis=1)
print(col_concat)

print(pd.concat([df1, df3], axis=1, join='inner'))
```

## 4.4 複数のデータセットをマージする

```python
person = pd.read_csv('data/survey_person.csv')
site = pd.read_csv('data/survey_site.csv')
survey = pd.read_csv('data/survey_survey.csv')
visited = pd.read_csv('data/survey_visited.csv')

print(person)
print(site)
print(visited)
print(survey)
```

### 4.4.1 1対1のマージ
- ある1列を他の1列に結合したいとき
- 結合したい2つの列に値の重複がないとき

```python
# site値の重複をなくしておく
visited_subset = visited.loc[[0,2,6], ]

# how のデフォルト値は 'inner' なので
# この場合は指定する必要がない
o2o_merge = site.merge(visited_subset, left_on='name', right_on='site')

print(o2o_merge)
```

### 4.4.2 多対1のマージ
- Many-to-One

```python
m2o_merge = site.merge(visited, left_on='name', right_on='site')
print(m2o_merge)
```

### 4.4.3 多対多のマージ
```python
ps = person.merge(survey, left_on='ident', right_on='person')
vs = visited.merge(survey, left_on='ident', right_on='taken')
print(ps)
print(vs)

ps_vs = ps.merge(vs,
  left_on=['ident', 'taken', 'quant', 'reading'],
  right_on=['person', 'ident', 'quant', 'reading'])

print(ps_vs.loc[0, ])
```
