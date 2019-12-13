# 3. ２次元データの整理


```python
import numpy as np
import pandas as pd

%precision 3
pd.set_option('precision', 3)

df = pd.read_csv('../data/ch2_xcores_em.csv', index_col='生徒番号')
```

```python
en_scores = np.array(df['英語'])[:10]
ma_scores = np.array(df['数学'])[:10]

scores_df = pd.DataFrame({
    '英語': en_scores,
    '数学': ma_scores
}, index=pd.Index(['A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J'], name='生徒'))
```

## 3.1 ２つのデータの関係性の指標
- **正の相関**: 例) 英語の点が高い人ほど数学の点も高い傾向にある
- **負の相関**: 例) 英語の点が高い人ほど数学の点は低い傾向にある
- **無相関**: 例） どちらも当てはまらず、英語の点数が数学の点数に直線的な影響を及ぼさない

### 3.1.1 共分散 covariance

```python
summary_df = scores_df.copy()
summary_df['英語の偏差'] =\
    summary_df['英語'] - summary_df['英語'].mean()
summary_df['数学の偏差'] =\
    summary_df['数学'] - summary_df['数学'].mean()
summary_df['偏差同士の積'] =\
    summary_df['英語の偏差'] * summary_df['数学の偏差']
summary_df
```

```python
summary_df['偏差同士の積'].mean()
```

> 共分散には *S<sub>xy</sub>* という表記がよく使われます<br>
> 今回の場合であれば変数*x*が英語, 変数*y* が数学に対応しています

NumPyの場合, 共分散は cov 関数で求めることができます。
ただし返り値は共分散という値ではなく, **共分散列（covariance matrix）** または **分散共分散行列（variance convariance matrix）**
と呼ばれる行列です。

```python
cov_mat = np.cov(en_scores, ma_scores, ddof=0)
dov_mat

# 1行目と1列目が第１引数の英語
# 2行目と2列目が第２引数の数学
# 1行2列目と2行1列目が英語と数学の共分散に該当します

cov_mat[0, 1], cov_mat[1, 0]
# 共分散
```

```python
# 同じ変数同士の共分散は分散と等しくなっています
cov_mat[0, 0], cov_mat[1, 1]

# 英語と数学の分散（上記と一致する）
np.var(en_scores, ddof=0), np.var(ma_scores, ddof=0)
```

### 3.1.2 相関係数


