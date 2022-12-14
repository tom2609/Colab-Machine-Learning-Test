import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")


def build_model():
  model = tf.keras.models.Sequential()


  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
  model.add(tf.keras.layers.Dropout(rate=0.2))
  model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model

def train_model(model, feature, label, epochs, batch_size):
  history = model.fit(x=feature, y=label, batch_size=batch_size, epochs=epochs)

  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  epochs = history.epoch

  hist = pd.DataFrame(history.history)

  loss = hist['accuracy']

  return trained_weight, trained_bias, epochs, loss

def plot_the_loss_curve(epochs, loss):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")

  plt.plot(epochs, loss, label="Loss")
  plt.legend()
  plt.ylim([loss.min()*0.97, loss.max()])
  plt.show()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 28, 28).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

epochs = 20
batch_size = 100

my_model = build_model()
trained_weight, trained_bias, epochs, loss = train_model(my_model, x_train, y_train, epochs, batch_size)
plot_the_loss_curve(epochs, loss)
my_model.save(r"C:\Users\Tom\Projects\Data Stash\mnist-model.h5", overwrite=True, include_optimizer=True)

selected_element = np.random.randint(0,9999)
test_feature = x_test[selected_element].reshape((28,28))
test_label = y_test[selected_element]
plt.imshow(test_feature, cmap="Greys")
plt.show()
print("Actual label: " + str(test_label))
print("Predicted label: " + str(my_model.predict(test_feature.reshape((1,28,28,1)))[0].argmax()))
