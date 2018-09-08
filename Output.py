import tensorflow.keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

test_x = pd.read_csv('./Data/test_x.csv')

model_name = '4 Adam'


model = keras.models.load_model('./Model_Analyse/{}.h5'.format(model_name))

test_y = model.predict(test_x)
test_y = np.exp(test_y)
test_y = pd.DataFrame(test_y, columns=['SalePrice'])
submission = pd.concat([pd.read_csv('./Data/test.csv')['Id'], test_y], axis=1)
submission.to_csv('./Predictions/{}.csv'.format(model_name), index=False)

sns.distplot(submission['SalePrice'])
plt.show()