from distutils.command.build import build
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

def build_model(my_learning_rate):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=10, input_shape=(1,)))
    model.add(tf.keras.layers.Dense(units=10, input_shape=(1,)))
    model.add(tf.keras.layers.Dense(units=10, input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate), loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

# def build_model(my_learning_rate):
#     model = tf.keras.models.Sequential()

#     model.add(tf.keras.layers.Dense(units=10))
#     model.add(tf.keras.layers.Dense(units=10))

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives()])

#     return model

def train_model(model, df, feature, label, epochs, batch_size):
    history = model.fit(x=df[feature], y=df[label], batch_size=batch_size, epochs=epochs)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    epochs = history.epoch

    hist = pd.DataFrame(history.history)

    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

print("Defined build_model and train_model")

def plot_the_model(trained_weight, trained_bias, feature, label):
    plt.xlabel("feature")
    plt.ylabel("label")

    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    plt.show()

def plot_the_loss_curve(epochs, rmse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions.")

def predict_house_values(n, feature, label):
  """Predict house values based on a feature."""

  batch = training_df[feature][10000:10000 + n]
  predicted_values = my_model.predict_on_batch(x=batch)

  print("feature   label          predicted")
  print("  value   value          value")
  print("          in thousand$   in thousand$")
  print("--------------------------------------")
  for i in range(n):
    print ("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + i], training_df[label][10000 + i], predicted_values[i][0] ))

training_df = pd.read_csv(r"C:\Users\Tom\Projects\Data Stash\california_housing_train.csv")

training_df["median_house_value"] /= 1000.0
training_df["rooms_per_person"] = training_df["total_rooms"] / training_df["population"]

learning_rate=0.001
epochs=30
my_batch_size=60

my_feature = "median_income"
my_label = "median_house_value"

my_model = None

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, training_df, my_feature, my_label, epochs, my_batch_size)

# my_model.summary()
# predict_house_values(10, my_feature, my_label)
# corr_matrix = training_df.corr()
# print(corr_matrix)
# plot_the_model(trained_weight, trained_bias, my_feature, my_label)
# plot_the_loss_curve(epochs, rmse)