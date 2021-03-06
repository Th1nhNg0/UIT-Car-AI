import random

import cv2
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle, random
import os
import matplotlib.pyplot as plt

MAX_ANGLE = 20
cutoff = 1500  # số lượng tối đa của một góc lái
filepath = "data/training_data-{}.npy"
joindata_path = "data/training_data_set.npy"

starting_value = 1
choose = [[] for _ in range(MAX_ANGLE * 2 + 1)]
while True:
    file_name = filepath.format(starting_value)
    if os.path.isfile(file_name):
        train_data = np.load(file_name, allow_pickle=True)
        shuffle(train_data)
        df = pd.DataFrame(train_data)
        # print(df.head())
        print(file_name, ":", Counter(df[1].apply(str)))
        for data in train_data:
            img = data[0]
            choice = data[1]
            choose[choice].append(data)
        starting_value += 1
    else:
        print('Stop at index:', starting_value)
        break

joindata = []
for i in range(MAX_ANGLE * 2 + 1):
    choose[i] = choose[i][:cutoff]
    joindata += choose[i]
df = pd.DataFrame(joindata)
print(Counter(df[1].apply(str)))
shuffle(joindata)
np.save(joindata_path, joindata)

totalFlip=0
for i in range(len(joindata)):
    if random()<0.05:
        joindata[i][0]=np.flip(joindata[i][0], 1)
        joindata[i][1] *= -1
        totalFlip+=1

print('totalFlip:',totalFlip)
print("Số dữ liệu sau khi balancing data:", len(joindata))
bins = np.arange(-MAX_ANGLE, MAX_ANGLE + 1.5) - 0.5
plt.hist([x[1] for x in joindata], density=False, bins=bins)
plt.show()
