import numpy as np
import cv2
import time
from alexnet import alexnet
from random import shuffle

WIDTH = 320
HEIGHT = 90
model = alexnet(WIDTH, HEIGHT, 1e-3)
MODEL_NAME = 'model/car-model'
LOAD_MODEL = False
EPOCHS = 10
filepath = "data/training_data-{}.npy"
train_data_file_count = 5

if LOAD_MODEL:
    model.load(MODEL_NAME)
    print('We have loaded a previous model!!!!')



for i in range(EPOCHS):
    data_order = [i for i in range(1, train_data_file_count + 1)]
    shuffle(data_order)
    for count, i in enumerate(data_order):
        file_name = filepath.format(i)
        training_data = np.load(file_name, allow_pickle=True)
        print('training_data-{}.npy'.format(i), len(training_data))
        train = training_data[:-int(len(training_data)*0.25)]
        test = training_data[-int(len(training_data)*0.25):]

        X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
        Y = [[0 if j != i[1] else 1 for j in range(-25, 26)] for i in train]
        test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
        test_y = [[0 if j != i[1] else 1 for j in range(-25, 26)] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    print('SAVING MODEL!')
    model.save(MODEL_NAME)
