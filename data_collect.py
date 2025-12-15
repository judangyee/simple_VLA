import pybullet as p
import pybullet_data
import numpy as np
import random
import cv2
import os
import time

# -----------------------------
# Basic setup
# -----------------------------
p.connect(p.GUI)  # 디버깅용 GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf", globalScaling=1)

#cube setting
if random.random() < 0.5:
    y_red = 0.3 + random.uniform(-0.05, 0.05)
    y_green = -0.3 + random.uniform(-0.05, 0.05)
else:
    y_red = -0.3 + random.uniform(-0.05, 0.05)
    y_green = 0.3 + random.uniform(-0.05, 0.05)

red_cube = p.loadURDF(
    "cube_small.urdf",
    basePosition=[0.5, y_red, 0.02]
)

green_cube = p.loadURDF(
    "cube_small.urdf",
    basePosition=[0.5, y_green, 0.02]
)

p.changeVisualShape(red_cube, -1, rgbaColor=[1, 0, 0, 1])
p.changeVisualShape(green_cube, -1, rgbaColor=[0, 1, 0, 1])

# -----------------------------
# Camera parameters
# -----------------------------
WIDTH, HEIGHT = 640, 480

view_matrix = p.computeViewMatrix(
    cameraEyePosition=[0.8, 0.0, 0.6],
    cameraTargetPosition=[0.5, 0.0, 0.0],
    cameraUpVector=[0, 0, 1]
)

proj_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=WIDTH / HEIGHT,
    nearVal=0.01,
    farVal=2.0
)

# -----------------------------
# Output directories
# -----------------------------
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/seg", exist_ok=True)

label_file = open("dataset/labels.txt", "w")

# -----------------------------
# Image capture
# -----------------------------
def capture(step, command, action):
    img = p.getCameraImage(
        WIDTH,
        HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.reshape(img[2], (HEIGHT, WIDTH, 4))[:, :, :3]
    rgb = rgb.astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"dataset/images/{step:06d}.png", rgb)

    seg = np.reshape(img[4], (HEIGHT, WIDTH)).astype(np.uint16)
    cv2.imwrite(f"dataset/seg/{step:06d}.png", seg)

    label_file.write(f"{step:06d}.png | {command} | {action}\n")

# -----------------------------
# Simple expert policy (rule-based)
# -----------------------------
commands = [
    ("pick the red cube", "MOVE_TO_RED"),
    ("pick the green cube", "MOVE_TO_GREEN")
]

step = 0
for command, action in commands:
    for _ in range(30):  # 같은 명령으로 여러 프레임 수집
        capture(step, command, action)
        p.stepSimulation()
        time.sleep(1. / 60.)
        step += 1

label_file.close()

print("Dataset capture finished. Press Ctrl+C to exit.")

# -----------------------------
# Keep GUI alive
# -----------------------------
while True:
    p.stepSimulation()
    time.sleep(1. / 240.)