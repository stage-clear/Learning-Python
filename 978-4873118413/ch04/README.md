# Matplotlib による可視化
- [ggplot2](https://github.com/tidyverse/ggplot2)
- [HoloViews](http://holoviews.org/)
- [Altair](https://altair-viz.github.io/)

## 4.1 Matplotlib の基礎知識
### 4.1.1 matplotlib のインポート

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
```

### 4.1.2 スタイルの設定

```python
plt.style.use('classic')
```

### 4.1.3 show() するか show() しないか？ 描画を表示する方法
#### 4.1.3.1 Python スクリプトからプロットする

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()
```

```shell
$ python myplot.py
```

#### 4.1.3.2 IPythonシェルからプロットする

```python
%matplotlib

import matplotlib.pyplot as plt
```

#### 4.1.3.3 IPython notebook からプロットする

```python
import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')
```

### 4.1.4 プロット結果のファイル保存

```python
fig.savefig('my_figure.png')
```

```python
from IPython.display import Image
Image('my_figure.png')
```

```python
fig.canvas.get_supported_filetypes()
```

## 4.2 同じ結果を得る２つのインターフェース

#### 4.2.0.1 MATLABスタイルフェース
MATLABスタイルのツールは pyplot (plt)インターフェースで提供されています

```python
# figureを作成する
plt.figure()

# 最初のグラフを作成し, 現在の座標軸に設定する
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))

# 2番目のグラフを作成して, 現在の座標軸に設定する
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
```

- `plt.gcf()` （get current figure: 現在の figure の取得）
- `plt.gca()` （get current axes: 現在の座標軸の取得）

#### 4.2.0.2 オブジェクト指向インターフェース

```python
# 最初にプロットのグリッドを作成する
# axは, 2つの座標軸オブジェクトの配列
fig, ax = plt.subplots(2)

# オブジェクトの plot() メソッドをコールする
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
```

## 4.3 単純な線グラフ

```python
import matplotlib as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

figure は, 軸, グラフィックス, テキスト, ラベルなど, すべてのオブジェクトを含む1つのコンテナと考えることができます.
axes は上に示した通り, 目盛とラベルを持つ境界です

```python
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))
```

```python
# pylabインターフェース
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
```

### 4.3.1 プロットの制御: 線の色とスタイル

```python
plt.plot(x, np.sin(x - 0), color='blue')
plt.plot(x, np.sin(x - 1), color='g')
plt.plot(x, np.sin(x - 2), color='0.75')
plt.plot(x, np.sin(x - 3), color='#ffdd44')
plt.plot(x, np.sin(x - 4), color=(1.0, 0.2, 0.3))
plt.plot(x, np.sin(x - 5), color='chartreuse')

# linestyle
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')

plt.plot(x, x + 4, linestyle='-')
plt.plot(x, x + 5, linestyle='--')
plt.plot(x, x + 6, linestyle='-.')
plt.plot(x, x + 7, linestyle=':')

plt.plot(x, x + 0, '-g') # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r') # dotted red
```

### 4.3.2 プロットの制御: 座標軸の範囲

```python
plt.plot(x, np.sin(x))

plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
```

軸を逆にしたい場合
```python
plt.plot(x, np.sin(x))

plt.xlim(10, 0)
plt.ylim(1.2, -1.2)
```

[plt.axis()](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.axis.html?highlight=axis#matplotlib.pyplot.axis)<br>
* axes は axis （座標軸）の複数形
```python
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5])
```

plt.axis() のさらに有益な使い方
```python
plt.plot(x, np.sin())
plt.axis('tight')
```

```python
plt.plot(x, np.sin(x))
plt.axis('equal')
```

### 4.3.3 プロットへのラベル付け

```python
plt.plot(x, np.sin(x))
plt.title('A Sine Curve')
plt.xlabel('x')
plt.ylabel('sin(x)')
```

```python
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')

plt.legend()
```

**Matplotlib雑学**

- `plt.xlabel()` → `ax.set_xlabel()`
- `plt.ylabel()` → `ax.set_ylabel()`
- `plt.xlim()` → `ax.set_xlim()`
- `plt.ylim()` → `ax.set_ylim()`
- `plt.title()` → `ax.set_title()`

```python
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(
    xlim=(0, 10),
    ylim=(-2, 2),
    xlabel='x',
    ylabel='sin(x)',
    title='A Simple Plot'
)
```

## 4.4 単純な散布図

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

### 4.4.1 plt.plot を使った散布図

```python
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black')
```

```python
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker, label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8)

plt.plot(x, y, '-ok') # line(-) circle marker(0), black(k)
```

```python
plt.plot(x, y, '-p', color='gray',
    markersize=15, linewidth=4,
    markerfacecolor='white',
    markeredgecolor='gray',
    markeredgewidth=2
)
plt.ylim(-1.2, 1.2)
```

[matplotlib.pyplot](https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.plot.html)

### 4.4.2 plt.scatter を使った散布図

```python
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
plt.colorbar()
```

```python
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], alpha=0.2,
    s=100 * features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
```

### 4.4.3 plot メソッド対scatterメソッド: 効率に関する注意点
データセットが数千ポイントを超えると、plt.plotはplt.scatterよりもずっと効率的です。

## 4.5 誤差の可視化
### 4.5.1 基本的なエラーバー

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k')
```

```python
plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
```

### 4.5.2 連続誤差

```python
from sklearn.gaussian_process import GaussianProcess
model = lambda x: x * np.sin(x)
xdata = np.array([1,3,5,6,8])
ydata = model(xdata)

gp = GaussianProcess(corr='cubic', theta0=13-2, thetaU=1E-1, random_start=100)
gp.fit(xdata[:, np.newaxis], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
dyfit = 2 * np.sqrt(MSE) # 2*σつまり95%の信頼区間

# 結果を可視化する
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')
plt.fil_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2)
plt.xlim(0, 10)
```

## 4.6 密度と等高線図

```python
% matplotlib inline 
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
```

### 4.6.1 ３次元関数の可視化

```python
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
```

等高線図は plt.contour関数を使用して作成できます

```python
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, colors='black')
```

```python
plt.contour(X, Y, Z, 20, cmap='RdGy')
```

```python
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
```

```python
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image')
```

```python
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5)
plt.colorbar()
```

## 4.7 ヒストグラム、ビニング、密度

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

data = np.random.randn(1000)

plt.hist(data)
```

```python
plt.hist(data, bins=30, normed=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
```

```phthon
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)
```

ヒストグラムの計算だけを行い、表示をしないのであれば、np.histogram()関数が利用できます
```python
counts, bin_edges = np.histogram(data, bins=5)
print(counts)
```

### 4.7.1 ２次元ヒストグラムとビニング

```python
# 多変量ガウス分布から得られたxおよびy配列のデータを定義します
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
```

#### 4.7.1.1 plt.hist2d: 2次元ヒストグラム

```python

```


