import numpy as np
import pandas as pd

train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')

data = train.append(test, ignore_index=True, sort=False)

# 删除两个在GrLivArea中奇怪的离散值
data = data.drop(data[data['Id'] == 1299].index)
data = data.drop(data[data['Id'] == 524].index)

# 删除Electrical
data = data.drop(data.loc[data['Electrical'].isnull()].index)

# SalePrice取对数
label = data.loc[:1459]['SalePrice']
label = np.log(label)

# 直接其他删除缺失值
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missingData = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data = data.drop((missingData[missingData['Total'] > 1]).index, 1)

# GrLivArea取对数
data['GrLivArea'] = np.log(data['GrLivArea'])
# 处理TotalBsmtSF
data['TotalBsmtSF'] = data['TotalBsmtSF'].map(lambda s: 0 if s == 0 else np.log(s))

# 创建所需数据的数组
smaller_train_x = pd.concat([data['OverallQual'], data['GrLivArea'], data['GarageCars'],
                          data['TotalBsmtSF'], data['FullBath'], data['YearBuilt']], axis=1)

smaller_train_x.loc[:1459].to_csv('./Data/train_x.csv', index=False)
smaller_train_x.loc[1460:].to_csv('./Data/test_x.csv', index=False)
label.to_csv('./Data/train_y.csv', index=False, header='SalePrice')