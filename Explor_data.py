import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import numpy as np

train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')
train['SalePrice'] = np.log(train['SalePrice'])

data = train.append(test, ignore_index=True, sort=True)
print(data['SalePrice'])
# data['SalePrice'] = np.log(data['SalePrice'])

# sns.distplot(data['KitchenAbvGr'])  # 绘制单一数据图
# plt.show()

# print(data['KitchenAbvGr'].describe())

# 绘制连续数值数据与目标数据的关系图
# var = 'GrLivArea'
# relation = pd.concat([data['SalePrice'], data[var]], axis=1)
# relation.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

# 绘制分类数据与目标数据的关系图（box）
# var = 'KitchenAbvGr'
# relation = pd.concat([data['SalePrice'], data[var]], axis=1)
# sns.boxplot(x=var, y='SalePrice', data=data)
# plt.show()

# 绘制Heatmap
# dataCorr = data.corr()
# sns.set(font_scale=0.6)
# sns.heatmap(dataCorr, linewidths=0.05,  cmap='RdBu', vmax=.8, square=True)
# plt.show()

# 绘制所需要的相关图
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(data[cols], height=2.5)
# plt.show()

# 缺失值
# total = data.isnull().sum().sort_values(ascending=False)
# # percent = (data.isnull().sum()/data.isnull().count() * 100).sort_values(ascending=False)
# # missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent (%)'])
# # print(missing_data.head(20))

# 绘制正态分布曲线
# fig, ax = plt.subplots(1, 1)
# sns.distplot(data['GrLivArea'], fit=norm)
# ax.set_title('GrLivArea(Original)')
# plt.show()

# 绘制正态概率图
# data['TotalBsmtSF'] = data['TotalBsmtSF'].map(lambda s: 0 if s == 0 else np.log(s))
# fig, ax = plt.subplots(1, 1)
# stats.probplot(data.loc[data['TotalBsmtSF'] != 0, 'TotalBsmtSF'], plot=plt)
# ax.set_title('TotalBsmtSF(log)')
# plt.show()

