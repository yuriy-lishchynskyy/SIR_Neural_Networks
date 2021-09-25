# tensorflow
import tensorflow as tf
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


cwd = os.getcwd()
out_subpath = "output_final"
out_file = "model_sir"
out_path = os.path.join(cwd, out_subpath, out_file)

# LOAD DATA
print("\nLoading Files")
data_raw = pd.read_csv('output_sir_monte_carlo_train.csv', delimiter=',', header=None)
data_train = data_raw.to_numpy()
print("\nTraining File Loaded")

indx_p_start = np.where(data_train[0] == 666)[0][0]
indx_i_start = np.where(data_train[0] == 777)[0][0]
indx_end = np.where(data_train[0] == 999)[0][0]

# SELECT DATA
n_days = 120

param_start = indx_p_start + 1
param_end = indx_i_start - indx_p_start

i_start = param_end + 1
i_end = i_start + n_days

y_data = data_train[:, param_start:param_end] # actual parameters value
y_data = np.delete(y_data, 2, axis=-1) # remove gamma info
x_raw = data_train[:, i_start:i_end]

# NORMALISE DATA
if False:
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_raw.T).T
else:
    N = 4500000
    x_data = x_raw / N

# SPLIT TRAINING/TEST DATA
tt_ratio = 0.8
st = len(x_data) # no. of total samples
st_split = int(tt_ratio * st)

x_train = x_data[:st_split, ]
x_test = x_data[st_split:, ]

y_train = y_data[:st_split, ]
y_test = y_data[st_split:, ]

print("\nTraining Data:")
print("X: {0}".format(x_train.shape))
print("Y: {0}".format(y_train.shape))

print("\nTesting Data:")
print("X: {0}".format(x_test.shape))
print("Y: {0}".format(y_test.shape))

n = len(x_train[0]) # length of input array
o = len(y_train[0]) # length of output array

# CREATE MODEL
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Input((n, )))
model1.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model1.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model1.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model1.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model1.add(tf.keras.layers.Dense(o, activation=tf.nn.sigmoid))

model1.summary()

# RUN MODEL
n_epoch = 20
n_batch = 32
model1.compile(optimizer='adam', loss='mae', metrics=[tf.keras.metrics.mean_absolute_percentage_error]) # model options
history1 = model1.fit(x_train, y_train, epochs=n_epoch, batch_size=n_batch, validation_data=(x_test, y_test), verbose=1) # run model

# SAVE MODEL
model1.save(out_path)

# PLOT LOSS
history_dict = history1.history
history_dict.keys()
train_acc = history_dict['mean_absolute_percentage_error']
test_acc = history_dict['val_mean_absolute_percentage_error']

epochs = np.linspace(1, n_epoch, num=n_epoch)

plt.figure()
plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
plt.plot(epochs, test_acc, 'r-', label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("MAPE")
plt.xticks(epochs)
plt.legend()
plt.show()

# TEST PREDICTIONS
y_test_hat = model1.predict(x_test, batch_size=n_batch) # make predictions on test set to compare with actual
test_loss, test_accuracy = model1.evaluate(x_test, y_test, batch_size=n_batch) # test dataset

print("\nTest Loss (MAE): {0}".format(test_loss))
print("\nTest Accuracy (MAPE): {0}".format(test_accuracy))

print("\nTest Data")
print("\nSample # \t n - Data \t\t n - NN \t\t Beta - Data \t\t Beta - NN")

for i in range(10): # compare first 10 testing samples/predictions
    print("{0} \t\t {1:.8f} \t\t {2:.8f} \t\t {3:.8f} \t\t {4:.8f}".format(i, y_test[i, 0], y_test_hat[i, 0], y_test[i, 1], y_test_hat[i, 1]))
