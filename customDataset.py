import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from modules.edgeDetection import crop_image
from numpy import save,load
import numpy as np

TRAIN_DATADIR = '/home/hariharan/Documents/Python Scripts/Animals/dataset/training_set'
TEST_DATADIR = '/home/hariharan/Documents/Python Scripts/Animals/dataset/test_set'
CATEGORIES = ['cats', 'dogs']

IMG_SIZE = 384

X = []
y = []

training_data = []
testing_data = []


def compute_hcf(x, y):  # find max of x or y and divide by 8,16,24 --> 16 for now
    max_dim = max(x, y)
    allowed_tile_sizes = [64, 32, 24, 16, 8]
    found = False
    i = 0
    while not found:
        n = max_dim % allowed_tile_sizes[i]
        if n == 0:
            return max_dim / allowed_tile_sizes[i]
        else:
            i = i + 1


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(TRAIN_DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:

                img_array = cv2.imread(os.path.join(path, img))  # crop, First resize respect aspect ratio, then convert to grayscale, then tile
                new_array = crop_image(img_array)
                img_height, img_width = new_array.shape
                # print(img_height,img_width)
                tile = compute_hcf(img_height, img_width)
                tile_height, tile_width = int(tile), int(tile)

                tiled_array = new_array.reshape(img_height // tile_height,
                                                tile_height,
                                                img_width // tile_width,
                                                tile_width)
                tiled_array = tiled_array.swapaxes(1, 2)
                # print(tiled_array)
                # new_array=cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([tiled_array, class_num])
            except Exception as e:
                pass
    return training_data


create_training_data()


def create_testing_data():
    for category in CATEGORIES:
        path = os.path.join(TEST_DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:

                img_array = cv2.imread(os.path.join(path, img))
                new_array = crop_image(img_array)
                img_height, img_width = new_array.shape
                # print(img_height,img_width)
                tile = compute_hcf(img_height, img_width)
                tile_height, tile_width = int(tile), int(tile)

                tiled_array = new_array.reshape(img_height // tile_height,
                                                tile_height,
                                                img_width // tile_width,
                                                tile_width)
                tiled_array = tiled_array.swapaxes(1, 2)
                # print(tiled_array.shape)
                # new_array=cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([tiled_array, class_num])
            except Exception as e:
                pass
    return testing_data


create_testing_data()

# print(training_data)
# print(len(testing_data))

np.save('training_data.npy', training_data)
np.save('testing_data.npy', testing_data)
# print(training_data)

# npx = numpy.array(training_data)
# print("Train Shape:", npx.shape)
# random.shuffle(training_data)

# for sample in training_data[:10]:
# 	print(sample[1])


for features, label in training_data:
    X.append(features)
    y.append(label)


# print(X)
# print(y)

# X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE, 1)

# print(X[1])
