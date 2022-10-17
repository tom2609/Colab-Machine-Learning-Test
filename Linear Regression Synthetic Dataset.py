import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def build_model(my_learning_rate):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=1, activation='relu', input_shape=(1,)))

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

print("Defined build_model and train_model")

def plot_the_model(trained_weight, trained_bias, feature, label):

  #achsen beschriften
  plt.xlabel("feature")
  plt.ylabel("label")

  # Plot the feature values vs. label values.
  plt.scatter(feature, label)

  # Create a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias
  x1 = feature[-1]
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')

  # Render the scatter plot and the red line.
  plt.show()

def plot_the_loss_curve(epochs, rmse):
  """Plot the loss curve, which shows loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions.")

# Define Features and Labels
my_feature = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

#Define Learning Rate and stuff
learning_rate=0.025
epochs=300
my_batch_size=20

my_model = build_model(learning_rate)

trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature, my_label, epochs, my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

# Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
# If the training loss does not converge, train for more epochs.
# If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
# If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
# Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
# Setting the batch size to a very small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
# For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.

#Reading the Dataset

#import pandas as pd
#from DRLearn import DRLearn

#titanic_dataset = pd.read_csv("titanic.csv", index_col=0)
#titanic_dataset.head()

#Exploring the dataset

#Plotting survival rate by class

#2 DRLearn.plot_passenger_class(titanic_dataset)

#Plotting survival rate by gender

#3 DRLearn.plot_passenger_gender(titanic_dataset)

#Preparing our data for the algorithm

#4 selected_features, target = DRLearn.extract_features(titanic_dataset)
#   selected_features.sample(5)

#Splitting our dataset into two parts: training and test

#5 X_train, X_test, y_train, y_test = DRLearn.split_dataset(selected_features, target, split=0.2)

#Training our model

#6 model = DRLearn.train_model(X_train, y_train)

#Evaluating the model

#7 DRLearn.evaluate_model(model, X_test, y_test)

#Analysing our model

#8 DRLearn.explain_model(model, X_train)

#9 model_interpretation = DRLearn.interpret_model(model, X_test, y_test)

#10 passenger_number = 3
#   DRLearn.analyze_passenger_prediction(model_interpretation, X_test, passenger_number)

#Understanding how the quantity of data affects our model

#11 DRLearn.visualise_training_progress(model, X_train, y_train, X_test, y_test)