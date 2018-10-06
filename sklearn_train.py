import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import AdaBoostRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
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

model_name = 'sklearn all model'

n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=7).get_n_splits(train_x.values)
    rmse = np.sqrt(-cross_val_score(model, train_x.values, train_y.values.ravel(), scoring="neg_mean_squared_error", cv=kf))
    return rmse


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=7))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=7))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
svr = SVR(kernel='poly', C=0.05)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=7)
model_Ada = AdaBoostRegressor(base_estimator=BayesianRidge(), n_estimators=1000)
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

# score = rmsle_cv(svr)
# print('SVR: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
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


class PolynomialModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, proportions):
        self.models = models
        self.proportions = proportions

    def fit(self, train_x, train_y):
        self.models_ = [clone(x) for x in self.models]
        self.proportions_ = [x for x in self.proportions]

        for model in self.models_:
            model.fit(train_x, train_y)
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X)*self.proportions_[i] for i, model in enumerate(self.models_)
        ])
        return np.sum(predictions, axis=1)


class StackingAverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=7)

        out_of_fold_prediction = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_prediction[holdout_index, i] = y_pred

        self.meta_model.fit(out_of_fold_prediction, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model.predict(meta_features)


stacked_models = StackingAverageModels(base_models=(KRR, ENet, GBoost), meta_model=lasso)
poly_model = PolynomialModels(models=(stacked_models, model_lgb, model_xgb), proportions=[0.7, 0.15, 0.15])
# poly_model.fit(train_x.values, train_y.values.ravel())
#
#
# poly_model.predict(test_x.values)


# stacked_models.fit(train_x.values, train_y.values.ravel())
# staked_train_pred = stacked_models.predict(train_x.values)
# stacked_pred = np.expm1(stacked_models.predict(test_x.values))
# print('Stacked: {}'.format(rmsle(train_y, staked_train_pred)))

score = rmsle_cv(poly_model)
print('stacked score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))

# model_xgb.fit(train_x.values, train_y.values.ravel())
# xgb_train_pred = model_xgb.predict(train_x.values)
# xgb_pred = np.expm1(model_xgb.predict(test_x.values))
# print('xgb: {}'.format(rmsle(train_y, xgb_train_pred)))

# score = rmsle_cv(model_xgb)
# print('xgb score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))

# model_lgb.fit(train_x.values, train_y.values.ravel())
# lgb_train_pred = model_lgb.predict(train_x.values)
# lgb_pred = np.expm1(model_lgb.predict(test_x.values))
# print('lgb: {}'.format(rmsle(train_y, lgb_train_pred)))

# score = rmsle_cv(model_lgb)
# print('lgb score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))

# average_all_model = StackingAverageModels(base_models=(model_lgb, model_xgb, stacked_models)
#                                           , meta_model=model_Ada)
# score = rmsle_cv(average_all_model)
# print('Average score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# stacked RanF score: 0.1139 (0.0057)
# average score: 0.1103 (0.0066)
# stack Ada score: 0.1105 (0.0067)
#
# print('All models:{}'.format(rmsle(train_y, staked_train_pred*0.7+xgb_train_pred*0.15+lgb_train_pred*0.15)))


# score = rmsle_cv(averaged_models)
# print('Stacking score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))


# test_y = stacked_pred * 0.7 + xgb_pred * 0.15 + lgb_pred * 0.15
# test_y = pd.DataFrame(test_y, columns=['SalePrice'])
# submission = pd.concat([pd.read_csv('./Data/test.csv')['Id'], test_y], axis=1)
# submission.to_csv('./Predictions/{}.csv'.format(model_name), index=False)
#
# sns.distplot(submission['SalePrice'])
# plt.show()