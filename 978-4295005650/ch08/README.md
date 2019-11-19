# 8. テキスト文字列の操作
## 8.1 はじめに
**目標**
1. 文字列の抽出
2. 文字列のメソッド
3. 文字列の整形
4. 正規表現

## 8.2 文字列

```python
word = 'grail'
sent = 'a scratch'
```

### 8.2 文字列の抽出とスライス
#### 8.2.1.1 １個の文字列を取得する

```python
print(word[0])

print(sent[0])
```

#### 8.2.1.2 複数の文字列をスライスする

```python
# 最初の3文字を取り出す
# index 3 は、4番目の文字であることに注意
print(word[0:3])
```

#### 8.2.1.3

```python
# 最後の文字を取得
print(word[-1])

# 'a' を取得
print(sent[-9:-8])

# 'a' を取得
print(sent[0:-8])

# 負のインデックスによるスライスでは、最後の文字を取得できない
print(sent[2:-1])
print(sent[-7:-1])
```

### 8.2.2 文字列の最後の文字を取得する

```python
# 最後の文字のインデックスは,
# len が返す数より1だけ小さい
s_len = len(sent)
print(s_len)

print(sent[2:s_len])
```

#### 8.2.2.1 先頭から、または末尾までスライスする

```python
print(word[0:3])
print(word[ :3])
```

```python
print(sent[2:len(sent)])
print(sent[2: ])
```

#### 8.2.2.2 スライスの増分

```python
print(sent[::2])
print(sent[::3])
```

## 8.3 文字列メソッド

```python
print(sent[::-1])
```

**Pythonの文字列メソッド**
|メソッド|説明|
|:-|:-|
|capitalize|最初の文字だけを大文字にする|
|count|部分文字列の出現回数を返す|
|startswith|文字列が指定の接頭辞(Prefix)で始まるなら True を返す|
|endswith|文字列が指定の接尾辞(Suffix)で終わるなら True を返す|
|find|部分文字列と最初にマッチしたインデックスを返す|
|index|findと同じだが、マッチしなければValueErrorを送出する|
|isalpha|すべての文字がアルファベットなら True を返す|
|isdecimal|すべての文字が数宇なら True を返す|
|isalnum|どの文字もアルファベットか数字なら True を返す|
|lower|アルファベットを全部小文字にしたコピーを返す|
|upper|アルファベットを全部大文字にしたコピーを返す|
|replace|ある文字列置換したものを返す|
|strip|デリミタ（区切り文字）で区切った値のリストを返す|
|partition|split(maxsplit=1)と同じだが、タプルとしてデリミタを含む値を返す|
|center|指定の長さでセンタリングした文字列を返す|
|zfill|指定の長さまで'0'を左詰めした文字列を返す|

### 8.5.1 フォーマットの形式
https://docs.python.org/ja/3.7/library/string.html#string-formating

### 8.5.2 文字列の書式化

```python
var = 'flesh wound'
s = 'It\'s just a {}!'

print(s.format(var))

print(s.format('scratch'))
```

```python
# 変数をインデックスで２回参照する
s = """Black Knight : 'Tis but a {0}.
King Arthur: A {0}? Your arm's off!
"""

print(s.format('scratch'))
```

```python
s = 'Hayden Planetarium Coordinates: lat, lon'

print(s.format(lat='40', lon="73"))
# *v3.7 で動作しなかった
```

###  8.5.3 数値の書式化

```python
print('Some digits of pi: {}'.format(3.14))
```

```python
print("In 2005, Lu Chao of China Recited {:,} digits of pi".format(67890))
# * {:, } スペースありだとエラー
```

```python
# {0:.4} と {0:.4%} の0は、この書式のインデックス0を示し,
# .4 は、小数点以下の桁数、４を示す
# 書式で % を使うと、少数がパーセントに書式化される
print("I remember {0:.4} or {0:.4%} of what Lu Chao recited".format(7/6890))
```

```
# 最初の0は、この書式のインデックス０を示し、
# 第２の0は、パディングに使う文字を示す.
# 次にある5は、合計の文字数であり、
# dは、10進数を使うという意味
# 文字列が５桁になるように0を使ってパディングする
print("My ID number is {0:05d}".format(42))
```

### 8.5.4 Cのprintfスタイルによる書式化

```python
# %dは10進の整数を表す
s = 'I only know %d digits of pi' % 7
print(s)

# sは文字列を表す
# 文字列パターンでは、角カッコ [] ではなく
# 丸カッコ () を使うことに注意
# 渡す変数は Python の dict なので、波カッコ {} を使う
print('Some digits of %(cont)s: %(value).2f' % {'cont': 'e', 'value': 2.718})
```

### 8.5.5 フォーマット済み文字列リテラル
f文字列

```python
var = 'flesh wound'
s = f"It's just a {var}!"

lat = '40.785 N'
lon = '73.973 W'
s = f"Hayden Planetarium Coordinates: {lat}, {lon}"
```

## 8.6 正規表現
https://regex101.com/

**基本的な正規表現の構文**
|構文|説明|
|:-|:-|
|.|どの１文字にもマッチする|
|^|文字列の先頭からマッチする|
|$|文字列の末尾からマッチする|
|\*|直前の文字の0回以上の最大の繰り返しとマッチする|
|+|直線の文字の1回以上の繰り返しとマッチする|
|?|直前の文字の0回または1回の繰り返しとマッチする|
|m|直前の文字のm回の繰り返しとマッチする|
|{m,n}|直前の文字のm回からn回までの最大の繰り返しとマッチする|
|¥|特殊文字をエスケープする|
|\[\]|文字集合（\[a-z\]は、aからzまでの文字）のどれかとマッチする|
|\||「または」の意味|
|()|丸カッコで囲まれたパターンと正確にマッチする|
|¥d|1桁の数字とマッチする（\[0-9\]）|
|¥D|数字ではない文字とマッチする(¥dの逆)|
|¥s|空白とマッチする|
|¥S|空白ではない文字とマッチする（¥sの逆）|
|¥w|ワード文字（英数文字とアンダースコア）にマッチする|
|¥W|ワード文字以外とマッチする（¥wの逆）|

**一般的な正規表現の関数**
|関数|説明|
|:-|:-|
|search|文字列で最初にパターンとマッチする場所を見つける|
|match|文字列の先頭からマッチを試みる|
|fullmatch|文字列全体とのマッチを試みる|
|split|パターンによって文字列を分割し、文字列のリストで返す|
|findall|重ならないすべてのマッチを見つけて、文字列のリストで返す|
|finditer|findallと似ているが、Pythonのイテレータを返す|
|sub|パターンにマッチしたものに対して、指定した文字列を置き換えたものを返す|

### 8.6.1 パターンとのマッチ
