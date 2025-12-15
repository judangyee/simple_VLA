from inference import infer
from executor import PyBulletExecutor
import time
import pybullet as p

executor = PyBulletExecutor()

image = "D:/Project/VLA_Project/dataset/images/000002.png"
while True:
    command = input("Enter Command (type 'end' to quit): ")

    if command.lower() == "end":
        print("Exiting...")
        break

    action = infer(image, command)
    print("Predicted:", action)

    time.sleep(0.5)
    executor.execute(action)
    time.sleep(0.5)  # 다음 명령 받기 전 잠깐 대기

while True:
    p.stepSimulation()
    time.sleep(1 / 240)  # 시뮬레이션 타임스텝과 동일

