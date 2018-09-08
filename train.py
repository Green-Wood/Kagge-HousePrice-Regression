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

model_name = '2 Adam model'

model = keras.Sequential([
    keras.layers.Dense(64, activation=keras.activations.relu, input_shape=(6,)),
    keras.layers.Dense(64, activation=keras.activations.relu),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam()
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=50, min_delta=0.001)

history = model.fit(train_x, train_y, epochs=50, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
model.save('./Model_Analyse/{}.h5'.format(model_name))

plot_history(history)
plt.show()
