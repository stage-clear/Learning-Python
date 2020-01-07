# 推測統計の基本
一部のデータから全体の統計的性質を推測する枠組みが, **推測統計** です

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%precision 3
%matplotlib inline
```

```python
df = pd.read_csv('../data/ch4_scores400.csv')
scores = np.array(df['点数'])
scores[:10]
```

## 4.1 母集団と標本
- 母集団 *population*
- 標本 *sample*
- 標本抽出 *sampling*
- 推定量 *estimator*
- 推定値 *estimate*

### 4.1.1 標本の抽出方法

> ランダムに標本を抽出する方法を**無作為抽出（random sampling）**といいます

> 複数回同じ標本を選ぶ抽出方法を**復元抽出(sampling with replacement)**、
> 同じ標本は一度しか選ばない方法を**非復元抽出(sampling without replacement)**といいます。




