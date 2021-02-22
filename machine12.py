import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import SimpleRNN,TimeDistributed,Embedding, Dense
from pprint import pprint

X = []
Y = []
for i in range(4):
    lst = list(range(i+1,i+4))
    X.append(list(map(lambda c: [c/10], lst)))
    Y.append(list(map(lambda c: [c/10+0.1], lst)))
X = np.array(X)
Y = np.array(Y)
X_test1 = np.array([[[0.5],[0.6],[0.7]]])
X_test2 = np.array([[[0.6],[0.7],[0.8]]])
model = Sequential()
model.add(SimpleRNN(100,input_shape=(3,1),return_sequences=True))
model.add(TimeDistributed(Dense(units=1)))
model.compile(loss='mse', optimizer='adam')
model.fit(X, Y,epochs=100)
model.summary()

print(model.predict(X_test1))
print(model.predict(X_test2))

model1 = Sequential()
model1.add(SimpleRNN(100,input_shape=(3,1),return_sequences=True))
model1.add(SimpleRNN(100,return_sequences=True))
model1.add(TimeDistributed(Dense(units=1)))
model1.compile(loss='mse', optimizer='adam')
model1.fit(X, Y,epochs=100)
model1.summary()

print(model1.predict(X_test1))
print(model1.predict(X_test2))
