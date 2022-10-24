import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import idx2numpy
import tensorflow as tf
from keras.datasets import mnist

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")


def build_model(my_learning_rate):
  model = tf.keras.models.Sequential()

  model.add(tf.keras.Input(shape=(784)))
  model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
  model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
  model.add(tf.keras.layers.Dense(units=10))

  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate), loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model

def train_model(model, df, feature, label, epochs, batch_size):
  history = model.fit(x=df[feature], y=df[label], batch_size=batch_size, epochs=epochs)

  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  epochs = history.epoch

  hist = pd.DataFrame(history.history)

  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse

def plot_the_loss_curve(epochs, rmse):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()

training_df = pd.read_csv(r"C:\Users\Tom\Projects\Data Stash\mnist_train.csv")

my_feature = "?"
my_label = "label"

my_learning_rate = 0.01
epochs = 10
batch_size = 20

my_model = build_model(my_learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, training_df, my_feature, my_label, epochs, batch_size)