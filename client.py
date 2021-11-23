# Import socket module
import socket
import sys
import time
import cv2
import numpy as np
from model import inception_v3

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
port = 54321

# connect to the server on local computer
s.connect(('host.docker.internal', PORT))

WIDTH = 160
HEIGHT = 60
MAX_ANGLE = 20

## edit các biến sau:
MODEL_NAME = 'model/car-colab-v0.7.model'
model = inception_v3(WIDTH, HEIGHT, 3, 1e-3, output=MAX_ANGLE*2+1, model_name=MODEL_NAME)
model.load(MODEL_NAME)


pre = time.time()
sendBack_angle = 0
sendBack_Speed = 0

def processImage(image):
    image = image[round(image.shape[0] / 3):, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    return image
try:
    print("START")
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
        image = processImage(image)
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
        prediction = model.predict([image.reshape(WIDTH, HEIGHT, 3)])[0]
        mode_choice = np.argmax(prediction)
        sendBack_angle = mode_choice - MAX_ANGLE
        # sendBack_angle = mode_choice - MAX_ANGLE
        sendBack_Speed = 30
        #print(sendBack_angle, prediction[mode_choice])
        # print(prediction)
        # cv2.imshow('window', image)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break


finally:
    print('closing socket')
    s.close()
