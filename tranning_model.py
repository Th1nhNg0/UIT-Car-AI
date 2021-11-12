import numpy as np
import cv2
import time
from model import alexnet
from random import shuffle

WIDTH = 160
HEIGHT = 60

# sửa các biến sau:
DATA_PATH = "data/training_data_set.npy"
MODEL_NAME = 'model/car-v0.model'
model = alexnet(WIDTH, HEIGHT, 1e-3)
EPOCHS = 10  # số lần train
test_percent = 0.2  # phần trăm tách ra từ training data để làm test data

LOAD_MODEL = False  # True nếu load model cũ để train tiếp
if LOAD_MODEL:
    model.load(MODEL_NAME)  # có thể đổi đường dẫn ở đây cho phù hợp
    print('We have loaded a previous model!!!!')

training_data = np.load(DATA_PATH, allow_pickle=True)

train = training_data[:-int(len(training_data) * test_percent)]
test = training_data[-int(len(training_data) * test_percent):]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [[0 if j != i[1] else 1 for j in range(-25, 26)] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [[0 if j != i[1] else 1 for j in range(-25, 26)] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
print('SAVING MODEL!')
model.save(MODEL_NAME)
