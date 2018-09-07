import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0:
        print('')
    print('.', end='')


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.ylim([0, 5])


train_x = pd.read_csv('./Data/train_x.csv')
train_y = pd.read_csv('./Data/train_y.csv')
test_x = pd.read_csv('./Data/test_x.csv')

model = keras.Sequential([
    keras.layers.Dense(64, activation=keras.activations.relu, input_shape=(train_x[1].shape,)),
    keras.layers.Dense(64, activation=keras.activations.relu),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.RMSprop()
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

history = model.fit(train_x, train_y, epochs=500, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
model.save('./Model_Analyse/1st original model')

plot_history(history)
plt.show()

test_y = model.predict(test_x)
submission = pd.concat([pd.read_csv('./Data/test.csv')['Id'], test_y], axis=1)
submission.to_csv('./Predictions/1st original model.csv')