import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

my_model = tf.keras.models.load_model(r"C:\Users\Tom\Projects\Data Stash\mnist-model.h5")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.reshape(10000, 28, 28).astype("float32") / 255

y_test = y_test.astype("float32")

selected_element = np.random.randint(0,9999)
test_feature = x_test[selected_element].reshape((28,28))
test_label = y_test[selected_element]
plt.imshow(test_feature, cmap="Greys")
plt.show()
print("Actual label: " + str(test_label))
print("Predicted label: " + str(my_model.predict(test_feature.reshape((1,28,28,1)))[0].argmax()))