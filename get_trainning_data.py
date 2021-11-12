# Import socket module
import socket
import keyboard
import os
import time
import cv2
import numpy as np

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
port = 54321

# connect to the server on local computer
s.connect(('127.0.0.1', port))
pre = time.time()


# Thay đổi các biến sau

MAX_ANGLE = 25  # góc bẻ lái tối đa từ 0 tới 25
angle_decrease_ratio=0.15 # tốc độ giảm về 0 của góc lái, 0.05 = giảm 5% theo thời gian
STABLE_SPEED = 30  # tốc độ mặc định
SAVE_CHECKPOINT = 1000 # mỗi 1000 dữ liệu ( ảnh + góc lái ) sẽ tự động lưu vào file, máy yếu lúc lưu có thể mất điều khiển xe
filepath = "data/training_data-{}.npy"


# không cần quan tâm  biến này
WIDTH = 160
HEIGHT = 60
sendBack_angle = 0
sendBack_Speed = 0


# file data
training_data = []
starting_value = 1
while True:
    file_name = filepath.format(starting_value)
    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)
        break
# đếm ngược
for i in list(range(2))[::-1]:
    print(i+1)
    time.sleep(1)
print("START COLLET DATA")

try:
    while True:
        message_getState = bytes("0", "utf-8")
        s.sendall(message_getState)
        state_date = s.recv(100)
        current_speed, current_angle = state_date.decode("utf-8").split(' ')
        message = bytes(f"1 {round(sendBack_angle)} {round(sendBack_Speed)}", "utf-8")
        s.sendall(message)
        data = s.recv(100000)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        cropimg = image[round(image.shape[0]/3):, :]
        # cropimg = cv2.cvtColor(cropimg, cv2.COLOR_RGB2YUV)
        cropimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)
        # cropimg = cv2.GaussianBlur(cropimg,(5,5),0)

        # bỏ comment phần này để xem hình ảnh mà xe thấy
        # cv2.imshow('window', cropimg)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

        cropimg = cv2.resize(cropimg, (WIDTH,HEIGHT))
        #cropimg = cropimg / 255

        training_data.append([cropimg, round(float(current_angle))])
        if len(training_data) % 100 == 0:
            print(len(training_data))
            if len(training_data) >= SAVE_CHECKPOINT:
                np.save(file_name, training_data)
                print('SAVED file index:',starting_value)
                training_data = []
                starting_value += 1
                file_name = filepath.format(starting_value)

        # nhận tính hiệu từ bàn phím để điều khiển góc lái, nếu không có tính hiệu thì giảm góc lái về 0
        if keyboard.is_pressed('a'):
            sendBack_angle -= 1
        elif keyboard.is_pressed('d'):
            sendBack_angle += 1
        else:
            sendBack_angle = sendBack_angle - sendBack_angle * angle_decrease_ratio
        sendBack_Speed = STABLE_SPEED
        # if keyboard.is_pressed('w'):
        #     sendBack_Speed += 5
        # if keyboard.is_pressed('s'):
        #     sendBack_Speed -= 5
        if keyboard.is_pressed('q'):
            break
        if abs(sendBack_angle) > MAX_ANGLE:
            sendBack_angle = MAX_ANGLE if sendBack_angle > 0 else -MAX_ANGLE
        if abs(sendBack_Speed) > 150:
            sendBack_Speed = 150 if sendBack_Speed > 0 else -150



finally:
    print('closing socket')
    s.close()
