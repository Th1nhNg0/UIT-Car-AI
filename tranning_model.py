import numpy as np
import cv2
import time
from model import inception_v3
from random import shuffle

WIDTH = 160
HEIGHT = 60

# sửa các biến sau:
MAX_ANGLE = 20
DATA_PATH = "data/training_data_set.npy"
MODEL_NAME = 'model/car-colab-v0.6.model'
model = inception_v3(WIDTH, HEIGHT, 3, 1e-3, output=MAX_ANGLE*2+1, model_name=MODEL_NAME)
EPOCHS = 5  # số lần train
test_percent = 0.2  # phần trăm tách ra từ training data để làm test data

LOAD_MODEL = False  # True nếu load model cũ để train tiếp
if LOAD_MODEL:
    model.load(MODEL_NAME)  # có thể đổi đường dẫn ở đây cho phù hợp
    print('We have loaded a previous model!!!!')

training_data = np.load(DATA_PATH, allow_pickle=True)

train = training_data[:-int(len(training_data) * test_percent)]
test = training_data[-int(len(training_data) * test_percent):]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
Y = [[0 if j != i[1] else 1 for j in range(-MAX_ANGLE, MAX_ANGLE+1)] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
test_y = [[0 if j != i[1] else 1 for j in range(-MAX_ANGLE, MAX_ANGLE+1)] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
print('SAVING MODEL!')
model.save(MODEL_NAME)
