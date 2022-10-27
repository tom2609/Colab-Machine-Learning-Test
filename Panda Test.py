import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from tensorflow import keras

# train_df = pd.read_csv(r"C:\Users\Tom\Projects\Data Stash\mnist_train.csv")

# check = random.randrange(0,60000)

# label = train_df.iat[check, 0]
# train_df = train_df.drop(labels='label', axis=1)
# train_df = train_df.iloc[check].values
# features = train_df.reshape((28,28))

# df = [label,features]

# np.set_printoptions(linewidth=150)
# print(df[1])


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 28, 28).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# def build_model(my_learning_rate):
#   model = tf.keras.models.Sequential()
  

#   model.add(tf.keras.Input(shape=(20, 28, 28)))
#   model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
#   model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
#   model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

#   model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate), loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])

#   return model

# def build_model(my_learning_rate):
#   inputs = tf.keras.Input(shape=(784,), name="digits")
#   x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
#   x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
#   outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

#   model = tf.keras.Model(inputs=inputs, outputs=outputs)

#   model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate), loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])

#   return model

# model.save('mnist-model/mnist-model.h5', overwrite=True, include_optimizer=True)

# saving the model
# save_dir = "/results/"
# model_name = 'keras_mnist.h5'
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)

# mnist_model = load_model(modelName)
# predicted_classes = mnist_model.predict_classes(X_test)

# # see which we predicted correctly and which not
# correct_indices = np.nonzero(predicted_classes == y_test)[0]
# incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
# print()
# print(len(correct_indices)," classified correctly")
# print(len(incorrect_indices)," classified incorrectly")

# # adapt figure size to accomodate 18 subplots
# plt.rcParams['figure.figsize'] = (7,14)

# figure_evaluation = plt.figure()

# # plot 9 correct predictions
# for i, correct in enumerate(correct_indices[:9]):
#     plt.subplot(6,3,i+1)
#     plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title(
#       "Predicted: {}, Truth: {}".format(predicted_classes[correct],
#                                         y_test[correct]))
#     plt.xticks([])
#     plt.yticks([])

# # plot 9 incorrect predictions
# for i, incorrect in enumerate(incorrect_indices[:9]):
#     plt.subplot(6,3,i+10)
#     plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title(
#       "Predicted {}, Truth: {}".format(predicted_classes[incorrect], 
#                                        y_test[incorrect]))
#     plt.xticks([])
#     plt.yticks([])

# figure_evaluation