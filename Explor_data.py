import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./Data/train.csv')

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
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data[cols], height=2.5)
plt.show()
