import numpy as np
import pandas as pd

data = pd.read_csv('./Data/train.csv')

# 删除两个在GrLivArea中奇怪的离散值
data = data.drop(data[data['Id'] == 1299].index)
data = data.drop(data[data['Id'] == 524].index)

# 直接删除缺失值
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missingData = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data = data.drop((missingData[missingData['Total'] > 1]).index, 1)
data = data.drop(data.loc[data['Electrical'].isnull()].index)

# SalePrice取对数
data['SalePrice'] = np.log(data['SalePrice'])
# GrLivArea取对数
data['GrLivArea'] = np.log(data['GrLivArea'])
# 处理TotalBsmtSF
data['TotalBsmtSF'] = data['TotalBsmtSF'].map(lambda s: 0 if s == 0 else np.log(s))

# 创建所需数据的数组
smaller_data = pd.concat([data['SalePrice'], data['OverallQual'], data['GrLivArea'], data['GarageCars'],
                          data['TotalBsmtSF'], data['FullBath'], data['YearBuilt']], axis=1)

smaller_data.to_csv('./Data/data_processed.csv')