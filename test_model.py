# Import socket module
import socket
import sys
import time
import cv2
import numpy as np
from alexnet import alexnet2

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
port = 54321

# connect to the server on local computer
s.connect(('127.0.0.1', port))

WIDTH = 320
HEIGHT = 90
model = alexnet2(WIDTH, HEIGHT, 1e-3)
MODEL_NAME = 'model/car-model'
model.load(MODEL_NAME)

pre = time.time()
sendBack_angle = 0
sendBack_Speed = 0




try:
    while True:
        # Send data
        message_getState = bytes("0", "utf-8")
        s.sendall(message_getState)
        state_date = s.recv(100)
        current_speed, current_angle = state_date.decode("utf-8").split(' ')
        message = bytes(f"1 {sendBack_angle} {sendBack_Speed}", "utf-8")
        s.sendall(message)
        data = s.recv(100000)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        """
        - Chương trình đưa cho bạn 1 giá trị đầu vào:
            * image: hình ảnh trả về từ xe

        - Bạn phải dựa vào giá trị đầu vào này để tính toán và gán lại góc lái và tốc độ xe vào 2 biến:
            * Biến điều khiển: sendBack_angle, sendBack_Speed
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """

        # your process here
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[image.shape[0] // 2:, :]
        image = cv2.resize(image, (WIDTH, HEIGHT))
        prediction = model.predict([image.reshape(WIDTH,HEIGHT,1)])[0]
        mode_choice = np.argmax(prediction)
        sendBack_angle = mode_choice-25
        sendBack_Speed = 35
        #print(sendBack_angle,prediction[mode_choice])
        print(prediction)
        cv2.imshow('window', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


finally:
    print('closing socket')
    s.close()