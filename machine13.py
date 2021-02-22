import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense

df_price = pd.read_csv('삼성전자-주가.csv')
seq_len = 50 #window 값이 50
sequence_length = seq_len + 1
high_prices = df_price['고가'].values
low_prices = df_price['저가'].values
mid_prices = (high_prices + low_prices)/2
result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index:index + sequence_length])
normalized_data = []
for window in result:
    normalized_window = [[(float(p) / float(window[0]))-1]for p in window]
    normalized_data.append(normalized_window)
result = np.array(normalized_data)
x_train = result[:8000,:-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = result[:8000, -1]
x_test = result[8000:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
y_test = result[8000:, -1]

model = Sequential()
model.add(LSTM(50,input_shape=(50,1),return_sequences=False) )
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=16)
#df = pd.DataFrame(y_test)
pred = model.predict(x_test)
plt.figure(figsize=(10,5))
plt.plot(y_test)
plt.plot(pred)


plt.show()