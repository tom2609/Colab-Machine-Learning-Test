import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import idx2numpy
import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

imagefile = r"C:\Users\Tom\Projects\Data Stash\train-images.idx3-ubyte"
imagearray = idx2numpy.convert_from_file(imagefile)

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



my_learning_rate = 0.01
epochs = 30
batch_size = 30

# plt.imshow(imagearray[3], cmap=plt.cm.binary)
# plt.show()

# print(imagearray[3])