import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
from numpy import save, load
import matplotlib.pyplot as plt
import time
import torch
# import cv2
import torch

# from google.colab import files
# from modules.customDataset import create_training_data
# from modules.customDataset import create_testing_data

train = load('modules/training_data.npy', allow_pickle=True)
test = load('modules/testing_data.npy', allow_pickle=True)

img_size = 320

x_train = []
y_train = []
x_test = []
y_test = []

# print(train)
for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

# print(x_train)
for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

print('Train Label:', y_train)
print('Test Label:', y_test)

x_train = np.array(x_train) / 255
x_train = np.reshape(x_train, (-1, 4))
# x_train = x_train.reshape(1700, 1, 1)


# x_train = torch.from_numpy(x_train)
# print(type(x_train))

x_test = np.array(x_test) / 255
# x_test = np.reshape(x_test, (-1, 4))

x_test = x_test.reshape(439, 1, 1)


y_train = np.array(y_train)
y_train = y_train.reshape(1700, 1, 1)

y_test = np.array(y_test)
y_test = y_test.reshape(439, 1, 1)

npx = numpy.array(x_train)
print("X Train Shape:", npx.shape)

npx_test = numpy.array(x_test)
print("X Test Shape:", npx_test.shape)

npy = numpy.array(y_train)
print("Y Train Shape:", npy.shape)

npy_test = numpy.array(y_test)
print("Y Test Shape:", npy_test.shape)

# print(y_test)
# reshape data to fit model
# x_train = x_train.reshape(16000, 1, 1, img_size, img_size,1)
# x_test = x_test.reshape(4000, 1, 1, img_size, img_size,1)

# index = 0
# Convert labels into set of numbers to input into neural network
# y_train_one_hot = to_categorical(y_train)
# y_test_one_hot = to_categorical(y_test)

# print("one hot label is: ", y_train_one_hot[index])

# x_train = x_train / 255
# x_test = x_test / 255
#
# Create the model's architecture
model = Sequential()

# Add first layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(1700, 1, 1), padding='same'))

# Add a pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Add another convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

# Add a pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Add a flattening layer
model.add(Flatten())

# Add a layer with 1000 neurons
model.add(Dense(1000, activation='relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer with 1000 neurons
model.add(Dense(500, activation='relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer with 1000 neurons
model.add(Dense(250, activation='relu'))

# Add a layer with 1000 neurons
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
start = time.time()
hist = model.fit(x_train, y_test,
                 batch_size=256,
                 epochs=10,
                 validation_split=0.2)

end = time.time()
print('time taken: ', end - start)

# torch.save(hist, 'mnist_model.pt')
model.save('modules/model.h5')

save('modules/x_test.npy', x_test)
save('modules/y_test_one_hot.npy', y_test)
