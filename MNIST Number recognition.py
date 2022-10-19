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

plt.imshow(imagearray[3], cmap=plt.cm.binary)
plt.show()

# print(imagearray[3])