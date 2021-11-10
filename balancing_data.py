import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import os

filepath = "data/training_data-{}.npy"
filepath_balanced = "data/training_data_balanced-{}.npy"

starting_value = 1
while True:
    file_name = filepath.format(starting_value)
    if os.path.isfile(file_name):
        train_data = np.load(file_name, allow_pickle=True)
        df = pd.DataFrame(train_data)
        #print(df.head())
        #print(Counter(df[1].apply(str)))
        shuffle(train_data)
        result = [[] for _ in range(51)]
        for data in train_data:
            img = data[0]
            choice = data[1]
            result[choice].append(data)

        saved_data = []
        count = sum(len(i) > 0 for i in result)
        for i in range(51):
            result[i] = result[i][:1000//count]
            saved_data += result[i]

        shuffle(saved_data)
        np.save(filepath_balanced.format(starting_value), saved_data)
        print('final file {} length:'.format(starting_value), len(saved_data))
        starting_value += 1
    else:
        print('Stop at index:', starting_value)
        break
