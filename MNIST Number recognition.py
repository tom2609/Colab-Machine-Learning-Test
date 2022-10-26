import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import idx2numpy
import tensorflow as tf
import random


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

def train_model(model, feature, label, epochs, batch_size):
  history = model.fit(x=feature, y=label, batch_size=batch_size, epochs=epochs)

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

imageFile = r"C:\Users\Tom\Projects\Data Stash\train-images.idx3-ubyte"
trainingFile = r"C:\Users\Tom\Projects\Data Stash\train-labels.idx1-ubyte"
trainingImage = idx2numpy.convert_from_file(imageFile)
trainingLabel = idx2numpy.convert_from_file(trainingFile)

check = random.randrange(0,60001)

image = trainingImage[check]
label = trainingLabel[check]

inputs = []
for rows_of_pixels in image:
	for pixel in rows_of_pixels:
		inputs.append(pixel)

my_learning_rate = 0.01
epochs = 30
batch_size = 20

my_model = build_model(my_learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, image, label, epochs, batch_size)