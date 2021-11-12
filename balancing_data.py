import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import os
import matplotlib.pyplot as plt

filepath = "data/training_data-{}.npy"
starting_value = 1
choose = [[] for _ in range(51)]
while True:
    file_name = filepath.format(starting_value)
    if os.path.isfile(file_name):
        train_data = np.load(file_name, allow_pickle=True)
        df = pd.DataFrame(train_data)
        # print(df.head())
        print(Counter(df[1].apply(str)))
        for data in train_data:
            img = data[0]
            choice = data[1]
            choose[choice].append(data)
        starting_value += 1
    else:
        print('Stop at index:', starting_value)
        break
big_data = []
for i in range(51):
    shuffle(choose[i])
    choose[i] = choose[i][:10000]
    big_data += choose[i]
df = pd.DataFrame(big_data)
print(Counter(df[1].apply(str)))
shuffle(big_data)
np.save('data/big_data_2', big_data)
print(len(big_data))
bins = np.arange(-25, 25 + 1.5) - 0.5
plt.hist([x[1] for x in big_data], density=False, bins=bins)  # density=False would make counts
plt.show()
