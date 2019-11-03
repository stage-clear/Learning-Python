# Learning-Python
```python
print("Hello, World!")
```
## 基本
### コメント
```python
# これはコメントです

"""
これは
複数行の
コメントです。
"""
```

### データ型
|型||
|:-:|:-:|
|文字列|str|
|整数|int|
|浮動少数点数|float|
|ブール値|bool|

### 変数
```python
x = 3
hi = "Hello, World"
my_float = 2.2
my_boolean = True
```

### 演算子
|演算子|意味|例|
|:-:|:-:|:-:|
|**|累乗|`2 ** 3`|
|%|割り算の余り|`14 % 4`|
|//|整数の割り算、切り捨て|`13 // 8`|
|/|割り算|`13 / 8`|
|\*|掛け算|`8 * 2`|
|-|引き算|`7 - 1`|
|+|足し算|`2 + 2`|

### 比較演算子
|演算子|意味|例|
|:-:|:-:|:-:|
|>|より大きい|`100 > 10`|
|<|より小さい|`100 < 10`|
|>=|以上|`2 >= 2`|
|<=|以下|`1 <= 4`|
|==|等価|`6 == 9`|
|!=|非等価|`3 != 2`|
|in|包含|`a in b`|

### 論理演算子
|演算子|意味|例|
|:-:|:-:|:-:|
|and|かつ|`True and True`|
|or|あるいは|`True or False`|
|not|否定|`not True`|

### 条件文
```python
home = "Japan"
if home == "USA":
  print("Hello, America!")
```
```python
x = 10
y = 11
if x == 10:
    if y == 11:
        print(x + y)
```
```python
home = "火星"
if home == "America":
    print("Hello, America!")
elif home == "Canada":
    print("Hello, Canada!")
elif home == "Thailand":
    print("Hello, Thailand!")
else:
    print("Hello, World")
```

### 関数
```python
def f(x):
    return x * 2

result = f(2)
print(result)
>> 4
```

### 組み込み関数
- https://docs.python.org/ja/3/library/functions.html

### スコープ
```python
x = 1
y = 2
z = 3

def f():
    print(x)
    print(y)
    print(z)
```

```python
def f():
    x = 1
    y = 2
    z = 3

print(x)
print(y)
print(z)
>> NameError: nama 'x' is not defined
```

```python
x = 100

def f():
    global x
    x += 1
    print(x)

f()
>> 101
```

### 例外処理
```python
a = input("type a number: ")
b = input("type another: ")
a = int(a)
b = int(b)
try:
    print(a / b)
except ZeroDivisionError:
    pritn("b cannot be zero.")
```

```python
try:
    a = input("type a number: ")
    b = input(type another: )
    a = int(a)
    b = int(b)
    print(a / b)
except (ZeroDIvisionError, ValueError):
    print("Invalid input.")
```

### ドキュメンテーション文字列
```python
def add (x, y):
    """
    Returns x + y.
    :param x: int.
    :param y: int.
    :return: int sum of x and y.
    """
    return x + y
```

### コンテナ
| |リスト|タプル|辞書|
|:-|:-|:-|:-|
|作成方法|`[]`|`()`|`{}`|
|データ構造|シーケンス（並び）|シーケンス（並び）|辞書|
|アクセス方法|変数[番号]|変数[番号]|変数[キー]|
|特徴|ミュータブル|イミュータブル|順序はなし|

### リスト
```python
fruit = list()
fruit = []
fruit = ["Apple", "Orange", "Pear"]
# リストの末尾に要素を追加
fruit.append("Banana")

# インデックスで要素を参照する
fruit[0]
fruit[1]
fruit[3]
>> Apple
>> Orange
>> Banana

# リストの末尾から要素を取り除く
item = fruit.pop()
item
>> Banana

# 足し算でリストを連結
colors1 = ["blue", "green", "yeloow"]
colors2 = ["orange", "pink", "black"]
colors1 + colors2
>> ["blue", "green", "yellow", "orange", "pink", "black"]

# 要素がリストに含まれているかどうか
colors = ["blue", "green", "yellow"]
"green" in colors
>> True
"black" not in colors
>> True

# リストのサイズを取得
len(colors)
>> 3
```

### タプル
- イミュータブル（変更不可能）なコンテナです
- 要素を追加することも変更することもできません

```python
my_tuple = tuple()
my_tuple = ()
my_tuple = (1, 2, 3)
my_tuple = (1,) # 要素が一つの場合は要素の直後にカンマ
```

### 辞書
- キーバリューペア
- あるバリューが辞書のバリューに使われているかどうかは確認できない
```python
my_dict = dict()
my_dict = {}
my_dict = {"Apple": "Red", "Banana": "Yellow"}

facts = dict()
# バリューを追加
facts["code"] = "fun"
# キーで参照
facts["code"]
>> fun

# キーバリューペアを削除
del facts["code"]
```

## 文字列の操作
三重クオート
```python
"""line one
   line two
   line three
"""
```
インデックス
```python
author = "Kafka"
author[0]
author[1]
author[2]
author[3]
author[-1]
```
文字列の足し算
```python
"Cat" + "in" + "hat"
```
文字列の掛け算
```python
"Sawyer" * 3
```
大文字小文字変換
```python
"We hold these truths...".upper()
"SO IT GOES.".lower()
"four score and...".capitalise()
```
書式化
```python
name = "ウィリアム・フォークナー"
"こんにちは、{}".format(name)
```
分割
```python
"水たまりを飛び越えたんだ。３メートルもあったんだぜ！".split("。")
```
結合
```python
first_three = "abc"
result = "+".join(first_three)
result
>> a+b+c

words = ["The", "fox", "jumped",
        "over", "the", "fence", "."]
one = "".join(words)
one
>> Thefoxjumpedoverthefence
```
空白除去
```python
s = "     The     "
s = s.strip()
s
>> The
```
置換
```python
equ = "All animals are equal."
equ = equ.replace("a", "@")
print(equ)
>> All @nim@ls @re equ@l.
```
文字を探す
```python
"animals".index("m")
>> 3

"animals".index("z")
>> ValueError: substring not found

try:
    "animals".index("z")
except:
    print("Not found.")
```
包含
```python
"Cat" in "Cat in the hat."
>> True
```
エスケープ文字
```python
"彼女は\"そうだね\"と言った"
```
スライス
```python
fict = ["トルストイ", "カミュ", "オーウェル", "ハクスリー", "オースティン"]
fict[0:3]
>>["トルストイ", "カミュ", "オーウェル"]

fict[2:] # ふたつめ行こう
fict[:] # 元のコピーを作る
```

## ループ
```python
name = "Ted"
for character in name:
    print(character)
```

```python
shows = ["GOT", "Narcos", "Vice"]
for show in shows:
    print(show)
```

```python
coms = ("A. Development", "Friends", "Always Sunny")
for show in coms:
    print(show)
```

```python
people = {"G. Bluth II": "A. Development", 
          "Barney": "HIMYM",
          "Dennis": "Always Sunny"}

for character in people:
    print(character)
```

```python
tv = ["GOT", "Narcos", "Vice"]
for i, new in enumerate(tv):
    new = tv[i]
    new = new.upper()
    tv[i] = new
```
range
```python
for i in range(1, 11):
    print(i)
>> 1
>> 9
...
>> 10
```
while
```python
x = 10
while x > 0:
    print('{}'.format(x))
    x -= 1
print("Happy New Year!")
```

```python
while True:
    print("Hello, World!")
```
break
```python
for i in range(0, 100):
    print(i)
    break
>> 0
```
continue
```python
for i in range(1, 6):
    if i == 3:
        continue
    print(i)

while i <= 5:
    if i == 3:
        i += 1
        continue
    print(i)
    i += 1
```
入れ子のループ
```python
# Sample1
for i in range(1, 3):
    print(i)
    for letter in ["a", "b", "c"]
        print(letter)

# Sample2
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
added = []
for i in list1:
    for j in list2:
        added.append(i + j)
print(added)

# Sample3
white input('y or n') != 'n':
    for i in range(1, 6):
        print(i)
```

## モジュール
- math
- random
- statistics
- keyword
- csv
- os

## ファイル
ファイルに書き出す
```python
import os
os.path.join("Users", "bob", "st.txt")
>> Uses/bob/st.txt
```

```python
st = open("st.txt", "w", "encoding="utf-8")
st.write("Hi from Python!")
st.close()
```
ファイルを自動的に閉じる
```python
with open("st.txt", "w") as f:
    f.write("Hi from Python!")
```
ファイルから読み込む
```python
with open("st.txt", "r", encoding="utf-8") as f:
    print(f.read())
```
