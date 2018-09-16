import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

train_x = pd.read_csv('./Data/train_x.csv')
train_y = pd.read_csv('./Data/train_y.csv')
test_x = pd.read_csv('./Data/test_x.csv')

model_name = 'sklearn average model'

n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=7).get_n_splits(train_x.values)
    rmse = np.sqrt(-cross_val_score(model, train_x.values, train_y.values.ravel(), scoring="neg_mean_squared_error", cv=kf))
    return rmse


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=7))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=7))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=7)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=True,
                             random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

# score = rmsle_cv(lasso)
# print('Lasso score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmsle_cv(ENet)
# print('ENet score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmsle_cv(KRR)
# print('KRR score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmsle_cv(GBoost)
# print('GBoost score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmsle_cv(model_xgb)
# print('XGB score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmsle_cv(model_lgb)
# print('LGB score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, train_x, train_y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(train_x, train_y)
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))


averaged_models.fit(train_x.values, train_y.values.ravel())
test_y = np.expm1(averaged_models.predict(test_x.values))
test_y = pd.DataFrame(test_y, columns=['SalePrice'])
submission = pd.concat([pd.read_csv('./Data/test.csv')['Id'], test_y], axis=1)
submission.to_csv('./Predictions/{}.csv'.format(model_name), index=False)

sns.distplot(submission['SalePrice'])
plt.show()