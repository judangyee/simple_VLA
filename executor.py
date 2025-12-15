import pybullet as p
import pybullet_data
import time
import numpy as np

class PyBulletExecutor:
    def __init__(self):
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1/240)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf")

        # Load Panda robot - 바닥에 배치
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            [0, 0, 0],
            useFixedBase=True
        )

        self.ee_link = 11  # Panda end-effector link index
        self.arm_joints = [0,1,2,3,4,5,6]

        # Load objects - 바닥 근처에 배치
        self.red_cube = p.loadURDF(
            "cube_small.urdf",
            [0.5, 0.2, 0.05],  # z 좌표를 낮춤 (바닥 근처)
            globalScaling=1,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        p.changeVisualShape(self.red_cube, -1, rgbaColor=[1, 0, 0, 1])

        self.green_cube = p.loadURDF(
            "cube_small.urdf",
            [0.5, -0.2, 0.05],  # z 좌표를 낮춤 (바닥 근처)
            globalScaling=1,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        p.changeVisualShape(self.green_cube, -1, rgbaColor=[0, 1, 0, 1])

        # Initial pose
        self.reset_arm()

    def reset_arm(self):
        # 바닥 작업에 적합한 자세
        init_pos = [0, 0.5, 0, -1.5, 0, 2.0, 0.785]
        for j, q in zip(self.arm_joints, init_pos):
            p.resetJointState(self.robot, j, q)

    def move_ee(self, target_pos, steps=500):
        joint_poses = p.calculateInverseKinematics(
            self.robot,
            self.ee_link,
            target_pos
        )

        for _ in range(steps):
            for j in self.arm_joints:
                p.setJointMotorControl2(
                    bodyIndex=self.robot,
                    jointIndex=j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_poses[j],
                    force=200
                )
            p.stepSimulation()
            time.sleep(1/240)

    def execute(self, action):
        if action == "MOVE_TO_RED":
            pos, _ = p.getBasePositionAndOrientation(self.red_cube)
            target = np.array(pos) + np.array([0, 0, 0.15])  # 오프셋 조정
            self.move_ee(target)

        elif action == "MOVE_TO_GREEN":
            pos, _ = p.getBasePositionAndOrientation(self.green_cube)
            target = np.array(pos) + np.array([0, 0, 0.15])  # 오프셋 조정
            self.move_ee(target)

        else:
            print(f"[WARN] Unknown action: {action}")