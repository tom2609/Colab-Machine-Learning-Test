import pandas as pd
import numpy as np
import idx2numpy
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical

# imageFile = r"C:\Users\Tom\Projects\Data Stash\train-images.idx3-ubyte"
# trainingFile = r"C:\Users\Tom\Projects\Data Stash\train-labels.idx1-ubyte"
# trainingImage = idx2numpy.convert_from_file(imageFile)
# trainingLabel = idx2numpy.convert_from_file(trainingFile)

# print(imageArray)
# print(trainingArray)





# training_df = pd.read_csv(r"C:\Users\Tom\Projects\Data Stash\california_housing_train.csv")
# yes = pd.read_csv(r"C:\Users\Tom\Projects\Data Stash\mnist_train.csv")

#print(training_df)
#print(yes)

# (trainX, trainy), (testX, testy) = mnist.load_data()

# print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
# print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

# print(trainingImage)
# print(trainingLabel)

# arr = []
# for ti in trainingImage:
#     for i in ti:
#         for e in i:
#             arr.append(e)

def load_dataset():
	# load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target value
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print(trainY[2])
    return trainX, trainY, testX, testY

def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# # evaluate model
		# _, acc = model.evaluate(testX, testY, verbose=0)
		# print('> %.3f' % (acc * 100.0))
		# # stores scores
		# scores.append(acc)
		# histories.append(history)
	return scores, histories

def run():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    return trainX, testX

trainX, testX = run()
#print(trainX)
# print(trainX[0])

#trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]