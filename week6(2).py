import keras
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
assert sys.version_info >= (3,5)
from tensorflow import keras
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
assert tf.__version__ >= "2.0"
from keras.layers.core import Activation,Dense
from keras.models import Sequential

#1
# w = 10
# b = 2
#
# X_train = np.array([[0],[1]])
# y_train = X_train*w+b
#
# X_test = np.array([[2],[2]])
# y_test = X_test*w+b
#
# model = Sequential()
#
# model.add(Dense(2,input_shape=(1,)))
# model.summary()
#
# model.compile(loss='mean_squared_error',optimizer='SGD')
# model_fit = model.fit(X_train,y_train,batch_size=3,epochs=1000,verbose=2)
#
# y_hat = model.predict(X_test)
# print(y_hat)
# print("Y_Test",y_test)

#2
# training_data = np.array([[0,0],[0,1],[1,0],[1,1]],"float32")
# target_data = np.array([[0],[1],[1],[0]],"float32")
#
# model = Sequential()
# model.add(Dense(32,input_dim=2,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(1,activation='relu'))
# model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])
# model.fit(training_data,target_data,epochs=1000,verbose=2)
# print(model.predict(training_data))

#3
