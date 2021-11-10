import numpy as np
import cv2
import time

#load file
training_data = np.load('training_data.npy',allow_pickle=True)
t=0.001 # thời gian show giữa các ảnh
index=2000 #_ vị trí của ảnh muốn xem, tự động tăng
while True:
    image,output=training_data[index]
    image = cv2.resize(image, (160*3, 45*3))
    index+=1
    processOutput = [0 for _ in range(-25, 26)]
    processOutput[output + 25] = 1
    print("Image {}, angle: {}".format(index,output))
    cv2.imshow('img', image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    time.sleep(t)