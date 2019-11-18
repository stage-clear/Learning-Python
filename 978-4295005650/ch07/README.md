# 7. データ型の概要と変換
## 7.1 はじめに
**目標**
1. DataFrame にある列のデータ型の判定
2. 各種のデータ型の相互変換
3. カテゴリ型データの使い方

## 7.2 データ型

```python
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')

print(tips.dtypes)
```

- int64 整数
- float64 浮動小数点数
- category カテゴリ型変数

## 7.3 型変換
### 7.3.1 文字列オブジェクトへの変換
値を文字列に変換するには、その列の `astype` メソッドを用いる

```python
tips['sex_str'] = tips['sex'].astype(str)

print(tips.dtypes)
```

### 7.3.2 数値への変換

```python
# total_bill を文字列に変換
tips['total_bill'] = tips['total_bill'].astype(str)

print(tips.dtypes)

# 再び float 型に変換
tips['total_bill'] = tips['total_bill'].astype(float)

print(tips.dtypes)
```

#### 7.3.2.1 to_numeric

```python
# tips データの抽出
tips_sub_miss = tips.head(10)

# 'missing' という値をいくつか代入する
tips_sub_miss.loc[[1,3,5,7], 'total_bill'] = 'missing'

print(tips_sub_miss)

# dtypes を見ると total_bill の列が文字列 object 型になっている
# 再び astype を使って float に戻そうとしたらエラーになるだろう
print(tips_sub_miss.dtypes)

# これはエラー
tips_sub_miss['total_bill'].astype(float)

# 代わりに pandas の to_numeric を使っても同様のエラーになる
pd.to_numeric(tips_sub_miss['total_bill'])

# errors=ignore' 列は何も変わらないが、エラーメッセージは出力されなくなる
tips_sub_miss['total_bill'] = pd.to_numeric(
  tips_sub_miss['total_bill'], errors='ignore')

# errors='coerce' の値を指定したら 'missing' の代わりに NaN が得られる
tips_sub_miss['total_bill'] = pd.to_numeric(
  tips_sub_miss['total_bill'], errors='coerce')
print(tips_sub_miss)
```

#### 7.3.2.2 to_numeric のダウンキャスト
to_numeric には、もう一つ downcast というパラメータがある
- integer
- signed
- unsigned
- float

```python
tips_sub_miss['total_bill'] = pd.to_numeric(
  tips_sub_miss['total_bill'],
  errors='coerce',
  downcast='float')

print(tips_sub_miss.dtypes)
```

## 7.4 カテゴリ型データ
1. この形式で保存すると、メモリと実行速度の効率がよくなる
2. カテゴリ型データは、値の列に順序が存在するとき（たとえばリッカーとのスケール）にも利用できる
3. 一部の Python ライブラリは（たとえ統計モデルの適合を行うとき）、カテゴリ型データの扱い方を知っている

### 7.4.1 カテゴリ型への変換
```python
# sex列を、まず文字列に変換する
tips['sex'] = tips['sex'].astype('str')
print(tips.info())

# sex列を、カテゴリ型のデータに戻す
tips['sex'] = tips['sex'].astype('category')
print(tips.info())
```

### 7.4.2 カテゴリ型データを操作する

**カテゴリ型のAPI**
|属性またはメソッド|説明|
|:-|:-|
|Series.cat.categories|このカテゴリ型のカテゴリを返す|
|Series.cat.ordered|カテゴリに順序があるかどうかを返す|
|Series.cat.codes|カテゴリの整数コードを返す|
|Series.cat.rename_categories()|カテゴリの名前を変更する|
|Series.cat.reorder_categories()|カテゴリの順序を変更する|
|Series.cat.add_categories()|新しいカテゴリを追加する|
|Series.cat.remove_categories()|カテゴリを削除する|
|Series.cat.remove_unused_categories()|使われていないカテゴリを削除する|
|Series.cat.set_categories()|新たにカテゴリを設定する|
|Series.cat.as_ordered()|カテゴリを順序つきにする|
|Series.cat.as_unordered()|カテゴリを順序なしにする|

## 7.5 まとめ
