# プロットによるグラフ描画
## 3.1 はじめに

```python
# アンスコムのデータセットは seaborn のライブラリにある
import seaborn as sns
anscombe = sns.load_dataset('anscombe')
print(anscombe)
```

**ssl.SSLCertVerificationError** への対処

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

## 3.2 matplotlib
```python
import matplotlib.pyplot as plt
```

```python
# データのサブセットを作る
# アンスコムの dataset の値がIとなるデータだけが含まれる
dataset_1 = anscombe[anscombe['dataset'] == 'I']

plt.plot(dataset_1['x'], dataset_1['y'])
```
> `plt.show()` を走れせないと表示されなかった

```python
# 円を描画したいときは `o` パラメータを渡す
plt.plot(dataset_1['x'], dataset_1['y'], 'o')
plt.show()
```

```python
# アンスコムのデータから残りのサブセットを作る
dataset_2 = anscombe[anscombe['dataset'] == 'II']
dataset_3 = anscombe[anscombe['dataset'] == 'III']
dataset_4 = anscombe[anscombe['dataset'] == 'IV']
```

```python
# サブプロットを入れる図の全体を作成する
fig = plt.figure()

# その図に、サブプロットの配置を伝える
# この例ではプロットを２行に並べ各行に２個ずつプロットを入れる

# サブプロット axes1 は2行2列のうち位置1に置く
axes1 = fig.add_subplot(2, 2, 1)

# サブプロット axes2 は2行2列のうち位置2に置く
axes2 = fig.add_subplot(2, 2, 2)

# サブプロット axes3 は2行2列のうち位置3に置く
axes3 = fig.add_subplot(2, 2, 3)

# サブプロット axes4 は2行2列のうち位置4に置く
axes4 = fig.add_subplot(2, 2, 4)

# さきほど作ったプロット領域のそれぞれにプロットを加える
axes1.plot(dataset_1['x'], dataset_1['y'], 'o')
axes2.plot(dataset_2['x'], dataset_2['y'], 'o')
axes3.plot(dataset_3['x'], dataset_3['y'], 'o')
axes4.plot(dataset_4['x'], dataset_4['y'], 'o')

# 個々のサブプロットに小さめのタイトルを追加
axes1.set_title('dataset_1')
axes2.set_title('dataset_2')
axes3.set_title('dataset_3')
axes4.set_title('dataset_4')

# 全体の図にタイトルを追加
fig.suptitle('Anscombe Data')

# タイトレイアウトを使う
fig.tight_layout()

fig.show()
```

## 3.3 matplotlib による統計的グラフィックス

```python
tips = sns.load_dataset('tips')
print(tips.head())
```

### 3.3.1 1変量データ
univariate
#### 3.3.1.1 ヒストグラム

```python
fig = plt.figure()
axes1 = fig.add_subplot(1, 1, 1)
axes1.hist(tips['total_bill'], bins=10)
axes1.set_title('Histogram of Total Bill')
axes1.set_xlabel('Frequency') # 度数
axes1.set_ylabel('Total Bill') # 総額
fig.show()
```

### 3.3.2 2変量データ
bivariate
#### 3.3.2.1 散布図

```python
scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1, 1, 1)
axes1.scatter(tips['total_bill'], tips['tip'])
axes1.set_title('Scatterplot of Total Bill vs Tip')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')
scatter_plot.show()
```

#### 3.3.2.2 箱ひげ図
boxplot
```python
boxplot = plt.figure()
axes1 = boxplot.add_subplot(1, 1, 1)
# boxplotの第１引数はデータである
# ここでは複数のデータをプロットするのからそれぞれのデータをリストに入れる必要がある
# それから、オプションのlabelsパラメータによって、渡しているデータにラベルを付ける
axes1.boxplot([tips[tips['sex'] == 'Female']['tip'], tips[tips['sex'] == 'Male']['tip']], labels=['Female', 'Male'])
axes1.set_xlabel('Sex') #性別
axes1.set_ylabel('Tip')
axes1.set_title('Boxplot of Tips by Sex')
boxplot.show()
```

### 3.3.3 多変量データ
multivariate

```python
# 性別に基づいた色の変数を作る
def recode_sex (sex):
    if sex == 'Female':
        return 0
    else:
        return 1
tips['sex_color'] = tips['sex'].apply(recode_sex)

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1, 1, 1)

axes1.scatter(
    x=tips['total_bill'],
    y=tips['tip'],
    s=tips['size'] * 10,
    c=tips['sex_color'],
    alpha=0.5
)
axes1.set_title('Total Bill vs Tip Colored by Sex and Sized by Size')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')
scatter_plot.show()
```

## 3.4 seaborn
```python
import seaborn as sns

tips = sns.load_dataset('tips')
```

### 3.4.1 1変量データ
#### 3.4.1.1 ヒストグラム

```python
# この subplots 関数は、個々のサブプロット用として
# 別々に作成した小さな図オブジェクト配列（ax）を、
# １個の図オブジェクト(hist)に追加する
# (matplotlib.pyplot.subplots)
hist, ax = plt.subplots()

# seaborn の distplot を使ってプロットを作成
ax = sns.distplot(tips['total_bill'])
ax.set_title('Total Bill Histogram with Density Plot')

plot.show() # 図を表示するのにも matplotlib.pyplot が必要
``` 

```python
# デフォルトでは、ヒストグラムとともにカーネル密度推定を使って密度もプロッティングする
# ヒストグラムだけほしいときは、kdeパラメータをFalseにセットする
hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], kde=False)
ax.set_title('Total Bill Histogram')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Frequency')
plt.show()
```

#### 3.4.1.2 密度プロット（KDE）
```python
den, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], hist=False)
ax.set_title('Total bill Density')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Unit Probablity') # 確率
plt.show()
```

#### 3.4.1.3 ラグプロット

```python
hist_den_rug, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], rug=True)
ax.set_title('Total Bill Histogram with Density and Rug Plot')
ax.set_xlabel('Total Bill')
plot.show()
```

#### 3.4.1.4 countplot（棒グラフ）

```python
count, ax = plt.subplots()
ax = sns.countplot('day', data=tips)
ax.set_title('Count of days')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Frequency')
plt.show()
```

### 3.4.2 2変量データ
#### 3.4.2.1 散布図

```python
scatter, ax = plt.subplots()
ax = sns.regplot(x='total_bill', y='tip', data=tips)
ax.set_title('Scatterplot of Total Bill and Tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
plt.show()
```

```python
joint = sns.jointplot(x='total_bill', y='tip', data=tips)
joint.set_axis_labels(xlabel='Total Bill', ylabel='Tip')

# タイトルを追加し、フォントサイズを設定し、
# テキストの位置を Total Bill の軸の上側に移動
joint.fig.suptitle('Joint Plot of Total Bill and Tip', fontsize=10, y=1.03)
```

#### 3.4.2.2 hexbin プロット（六角形ビニング）
```python
hexbin = sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')
hexbin.set_axis_labels(xlabel='Total Bill', ylabel='Tip')
hexbin.fig.suptitle('Hexbin Joint Plot of Total Bill and Tip', fontsize=10, y=1.03)
```

#### 3.4.2.3 ２次元の密度プロット
```python
kde, ax = plt.subplots()
ax = sns.kdeplot(data=tips['total_bill'], data2=tips['tip'], shade=True)
ax.set_title('Kernel Density Plot of Total Bill and Tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
plt.show()

kde_joint = sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')
```

#### 3.4.2.4 棒グラフ

```python
bar, ax = plt.subplots()
ax = sns.barplot(x='time', y='total_bill', data=tips)
ax.set_title('Bar plot of average total bll for time of day')
ax.set_xlabel('Time of day')
ax.set_ylabel('Average total bill')
plt.show()
```

#### 3.4.2.5 箱ひげ図

```python
box, ax = plt.subplots()
ax = sns.boxplot(x='time', y='total_bill', data=tips)
ax.set_title('Boxplot of total bill by time of day')
ax.set_xlabel('Time of day')
ax.set_ylabel('Total Bill')
plt.show()
```

#### 3.4.2.6 バイオリンプロット

```python
violin, ax = plt.subplots()
ax = sns.violinplot(x='time', y='total_bill', data=tips)
ax.set_title('Violin plot of total bill by time of day')
ax.set_xlabel('Time of day')
ax.set_ylabel('Total Bill')
plt.show()
```

#### 3.4.2.7 ペアの相関を見る
```python
fig = sns.pairplot(tips)
```

```python
pair_grid = sns.PairGrid(tips)
# sns.regplot の代わりにplt.scatterを使うことも可能
pair_grid = pair_grid.map_upper(sns.regplot)
pair_grid = pair_grid.map_lower(sns.kdeplot)
pair_grid = pair_grid.map_diag(sns.distplot, rug=True)
plt.show()
```

### 3.4.3 多変量データ
#### 3.4.3.1 色の違い

```python
violin, ax = plt.subplots()
ax = sns.violinplot(x='time', y='total_bill', hue='sex', data=tips, split=True)
plt.show()
```

```python
# ここでは regplot ではなく lmplot を使っていることに注意
scatter = sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', fit_reg=False)
plt.show()
```

```python
fig = sns.pairplot(tips, hue='sex')
```

#### 3.2.3.2 サイズと形
```python
scatter = sns.lmplot(x='total_bill', y='tip', data=tips, 
    fit_reg=False, hue='sex', 
    scatter_kws={'s': tips['size'] * 10}
)

plt.show()
```

```python
scatter = sns.lmplot(x='total_bill', y='tip', data=tips,
    fit_reg=False, hue='sex', markers=['o', 'x'],
    scatter_kws={'s': tips['size'] * 10})
```

#### 3.4.3.3 ファセット（切り口）

```python
anscombe_plot = sns.lmplot(x='x', y='y', data=anscombe, fit_reg=False, col='dataset', col_wrap=2)
plt.show()
```

```python
# FacetGrid を作る
facet = sns.FacetGrid(tips, col='time')
# time のそれぞれの値について、総額のヒストグラムをプロット
facet.map(sns.distplot, 'total_bill', rug=True)
plt.show()
```

```python
facet = sns.FacetGrid(tips, col='day', hue='sex')
facet = facet.map(plt.scatter, 'total_bill', 'tip')
facet = facet.add_legend()
plt.show()
```

```python
fig = sns.lmplot(x='total_bill', y='tip', data=tips, fit_reg=False, hue='sex', col='day')
plt.show()
```

```python
facet = sns.FacetGrid(tips, col='time', row='smoker', hue='sex')
facet.map(plt.scatter, 'total_bill', 'tip')
```

```python
facet = sns.catplot(x='day', y='total_bill', hue='sex', data=tips, row='smoker', col='time', kind='violin')
```

## 3.5 pandasのオブジェクト
### 3.5.1 ヒストグラム
- `Series.plot.hist()`
- `DateFrame.plot.hist()`

```python
# Seriesのヒストグラム
fig, ax = plt.subplots()
ax = tips['total_bill'].plot.hist()
plt.show()
```

```python
# DataFrame のヒストグラム
# 棒が重複しても透けて見えるように、alpha で透明度を設定
fig, ax = plt.subplots()
ax = tips[['total_bill', 'tip']].plot.hist(alpha=0.5, bins=20, ax=ax)
plt.show()
```

### 3.5.2 密度プロット
```python
fig, ax = plt.subplots()
ax = tips['tip'].plot.kde()
plt.show()
```

### 3.5.3 散布図
```python
fig, ax = plt.subplots()
ax = tips.plot.scatter(x='total_bill', y='tip', ax=ax)
plt.show()
```

### 3.5.4 hexbin プロット
```python
fig, ax = plt.subplots()
ax = tips.plot.hexbin(x='total_bill', y='tip', ax=ax)
plt.show()
```

```python
fig, ax = plt.subplots()
ax = tips.plot.hexbin(x='total_bill', y='tip', gridsize=10, ax=ax)
plt.show()
```

### 3.5.5 箱ひげ図
```python
fig, ax = plt.subplots()
ax = tips.plot.box(ax=ax)
plt.show()
```

## 3.6 seabornのテーマとスタイル
- darkgrid
- whitegrid
- dark
- white
- ticks

```python
fig, ax = plt.subplots()
ax = sns.violinplot(x='time', y='total_bill', hue='sex', data=tips, split=True)
plt.show()
```

```python
sns.set_style('whitegrid')
fig, ax = plt.subplots()
ax = sns.violinplot(x='time', y='total_bill', hue='sex', data=tips, split=True)
plt.show()
```

```python
fig = plt.figure()
seaborn_styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
for idx, style in enumerate(seaborn_styles):
    plot_position = idx + 1
    with sns.axes_style(style):
        ax = fig.add_subplot(2, 3, plot_position)
        violin = sns.violinplot(x='time', y='total_bill', data=tips, ax=ax)
        violin.set_title(style)
fig.tight_layout()
plt.show()
```

