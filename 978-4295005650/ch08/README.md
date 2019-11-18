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
