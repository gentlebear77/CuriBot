import pybullet as p
import numpy as np
from dinobot.robot_setup import get_gripper_position, make_robot
from dinobot.robot_control import open_gripper, close_gripper
from dinobot.environment_setup import get_object_position, get_object_velocity
import dinobot.pybullet_tools.utils as pb_utils
import time
import math
import cv2
import matplotlib.pyplot as plt


class SimpleEnv:
    def __init__(self, Test_env=False):

        self.robot_urdf = "/home/gentlebear/Mres/dinobot/assets/franka_description/robots/franka_panda.urdf"
        self.robot = None
        self.ball = None
        self.target = None
        self.objects = []
        self.joint_id = []
        self.client = None
        self.test = Test_env
        
        self.position_control_gain_p = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self.position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        self.max_torque = [87, 87, 87, 87, 12, 12, 12, 35, 35]   # from website
        self.gripper_joints = [8, 9]
        self.gripper_open_pos = 0.04
        self.gripper_close_pos = 0.015
        self.setup_env()
        self.dof = p.getNumJoints(self.robot, physicsClientId=self.client) - 1
        self.robot_data = {}
        print("Number of dof: ", self.dof)
        for j in range(self.dof):
            joint_info = p.getJointInfo(self.robot, j)
            # print(joint_info)
            self.joint_id.append(j)
            # self.q_min.append(joint_info[8])
            # self.q_max.append(joint_info[9])
            # self.target_pos.append((self.q_min[j] + self.q_max[j])/2.0)
            # self.target_torque.append(0.)
            self.robot_data[f"joint_{j}"] = {
                "name": joint_info[1].decode('utf-8'),
                "type": joint_info[2],
                "lower_limit": joint_info[8],
                "upper_limit": joint_info[9],
                "max_force": joint_info[10],
                "max_velocity": joint_info[11],
            }
        # print("Joint names: ", [self.robot_data[f"joint_{j}"]["name"] for j in range(self.dof)])

        initial_value, inital_velocity = self.getJointStates()

        joint_states = p.getJointStates(self.robot, self.joint_id)
        self.initial_torque = [x[3] for x in joint_states]
        # print("Initial torque: ", self.initial_torque)
        # print("Initial velocity: ", inital_velocity)
        # print("Initial value: ", initial_value)
        self.initial_value = initial_value
        self.initial_velocity = inital_velocity
        # self.IKmodel = IK.IKSolver(self.robot_urdf, "panda_hand", [0, 0, 0, 0], np.eye(4))

    def setup_env(self):

        self.client = p.connect(p.GUI)
        # import pybullet_data
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("/home/gentlebear/Mres/dinobot/assets/plane/plane.urdf")
        p.setTimeStep(1 / 1000, physicsClientId=self.client)
        p.setPhysicsEngineParameter(solverResidualThreshold=0, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=200)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client)
        robot_idx = p.loadURDF(self.robot_urdf, useFixedBase=True, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client)

        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=80, cameraPitch=-30, cameraTargetPosition=[0, 0, 0], physicsClientId=self.client)
        # set_robot_to_reasonable_position(robot_idx)
        p.changeDynamics(robot_idx, 8, linearDamping=0, lateralFriction=1, physicsClientId=self.client)
        p.changeDynamics(robot_idx, 9, linearDamping=0, lateralFriction=1, physicsClientId=self.client)

        reasonable_joint_numbers = list(range(0,7))
        reasonable_joint_positions = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]

        for joint, value in zip(reasonable_joint_numbers, reasonable_joint_positions):
            p.resetJointState(robot_idx, joint, targetValue=value, targetVelocity=0, physicsClientId=self.client)

        self.robot = robot_idx
        if not self.test:
            self.load_urdf_object("/home/gentlebear/Mres/dinobot/assets/pybullet_object_models/cube/red_cube.urdf", position=(0.5, 0, 0.1))
        else:
            self.load_urdf_object("/home/gentlebear/Mres/dinobot/assets/pybullet_object_models/cube/blue_cube.urdf", position=(0.6, 0, 0.1))
            # self.load_urdf_object("/home/gentlebear/Mres/dinobot/assets/pybullet_object_models/ycb_objects/YcbPottedMeatCan/model.urdf", position=(0.5, 0.1, 0.1), globalScale=0.8)
            # self.load_urdf_object("/home/gentlebear/Mres/dinobot/assets/pybullet_object_models/ycb_objects/YcbBanana/model.urdf", position=(0.6, 0.1, 0.1))
            # self.load_urdf_object("/home/gentlebear/Mres/dinobot/assets/pybullet_object_models/ycb_objects/YcbPowerDrill/model.urdf", position=(0.5, 0.1, 0.1), globalScale=0.7)
            # self.load_urdf_object("/home/gentlebear/Mres/dinobot/assets/pybullet_object_models/cube/red_cube.urdf", position=(0.35, 0, 0.55), globalScale=0.5)
        # disable shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        # self.reset_wrist_camera()

        img, depth = self.get_wrist_camera_image()
        
        cv2.imwrite('rgb_bn.png', img)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        # initial env
        for _ in range(200):
            self.step()
            # time.sleep(1 / 1000)

    def load_urdf_object(self, urdf_path, position=(0, 0, 0), orientation=(0, 0, 0, 1), globalScale=0.8):
        """
        Load a URDF model and add it to the environment.
        
        Args:
            urdf_path (str): Path to the URDF file.
            position (tuple): Initial position of the object (x, y, z).
            orientation (tuple): Initial orientation as a quaternion (x, y, z, w).
        """
        object_id = p.loadURDF(urdf_path, basePosition=position, baseOrientation=orientation, globalScaling=globalScale)
        self.objects.append(object_id)
        return object_id

    def reset(self):

        for joint in self.joint_id:
            p.resetJointState(self.robot, joint, targetValue = self.initial_value[joint], targetVelocity = 0)

        p.setJointMotorControlArray(
            self.robot, 
            self.joint_id, 
            controlMode=p.POSITION_CONTROL,
            forces=self.max_torque
        )

        for _ in range(10):
            self.step()
            # time.sleep(1 / 1000)
        

    def setTargetPositions(self, target_joint_positions):
        """
        Set the target joint positions for the robot.
        
        Args:
            target_joint_positions (list): List of target joint positions.
        """
        # print("joint_id: ", self.joint_id)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joint_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=target_joint_positions,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def setTargetVelocities(self, target_joint_velocities):
        """
        Set the target joint velocities for the robot.
        
        Args:
            target_joint_velocities (list): List of target joint velocities.
        """
        # print("Setting target velocities: ", target_joint_velocities)

        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joint_id,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=target_joint_velocities,
                                    forces=self.max_torque) 
        
    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joint_id,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.target_torque)
        
    def setEndEffectorPose(self, target_position, target_orientation):
        target_joint_positions = self.InverseKinematics(target_position, target_orientation)
        self.setTargetPositions(target_joint_positions)
        
    def setDeltaEndControl(self, delta_position, delta_orientation):
        # print("Delta position: ", delta_position)
        current_position, current_orientation = self.getEndEffectorPose()
        target_position = np.array(current_position) + np.array(delta_position)
        target_orientation = p.multiplyTransforms([0, 0, 0], current_orientation, [0, 0, 0], delta_orientation)[1]
        # target_orientation =  np.array(current_orientation) + np.array(delta_orientation)
        target_joint_positions = self.InverseKinematics(target_position, current_orientation)
        # current_joint_positions, _ = self.getJointStates()
        # target_joint_positions = current_joint_positions + delta_position
        # print("Target joint positions: ", target_joint_positions)
        self.setTargetPositions(target_joint_positions)
    
    def setDeltaPositionControl(self, delta_position):
        
        current_joint_positions, _ = self.getJointStates()
        target_joint_positions = current_joint_positions + delta_position
        # print("Target joint positions: ", target_joint_positions)
        self.setTargetPositions(target_joint_positions)
    

    def InverseKinematics(self, target_position, target_orientation):
        """
        Compute the inverse kinematics solution for the robot to reach the target position and orientation.
        
        Args:
            target_position (list): Target position (x, y, z).
            target_orientation (list): Target orientation as a quaternion (x, y, z, w).
        
        Returns:
            list: List of target
        """
        # print("Target position: ", target_position)
        # print("Target orientation: ", target_orientation)
        return list(p.calculateInverseKinematics(self.robot, 7, target_position, target_orientation, residualThreshold=1e-5, maxNumIterations=1000))

    # use robot_control to move the robot to the target position

    def control_gripper(self, gripper):
        if gripper:
            open_gripper(self.robot)
        else:
            close_gripper(self.robot)
    
        
    def reset_velocities(self, target_value=None):
        for joint in self.joint_id:
            p.resetJointState(self.robot, joint, targetValue = target_value[joint], targetVelocity = 0)
        
        p.setJointMotorControlArray(
            self.robot, 
            self.joint_id, 
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0 for _ in self.joint_id],
            forces=self.max_torque
        )
        for _ in range(100):
            self.step()
            # time.sleep(1 / 1000)

        
    def controlGripper(self, gripper):
        target_pos = self.gripper_open_pos if gripper else self.gripper_close_pos
        # print("self.gripper_joints: ", self.gripper_joints)
        p.setJointMotorControlArray(bodyUniqueId=self.robot, 
                                    jointIndices=self.gripper_joints,
                                    controlMode=p.POSITION_CONTROL, 
                                    targetPositions=[target_pos, target_pos], 
                                    forces=[5*240., 5*240.])

        # time.sleep(1 / 1000)
        # p.setJointMotorControl2(self.robot, self.gripper_joints[1], p.POSITION_CONTROL, targetPosition=target_pos)


    def getEndEffectorPose(self):
        link_state = p.getLinkState(self.robot, 7)  # Link 9 corresponds to the end-effector in Panda URDF
        ee_pos = link_state[4]  # End-effector position
        ee_ori = link_state[5]  # End-effector orientation in quaternion
        return ee_pos, ee_ori

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.joint_id)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel 
    
    # def reset_wrist_camera(self):
    #     # Reset the wrist camera to its initial position.
    #     wrist_position = get_gripper_position(self.robot)
    #     print("End position: ", wrist_position)
    #     print(self.getEndEffectorPose)
    #     wrist_orientation = [1, 0, 0, 0]  # Default orientation (no rotation)
    #     forward_vector = np.array([0, 0, 1])
    #     rotation_matrix = np.array(p.getMatrixFromQuaternion(wrist_orientation)).reshape(3, 3)
    #     position_offset = np.array([0.05, 0, 0.05])
    #     print("End position: ", wrist_position)
    #     wrist_position = np.array(wrist_position) + position_offset
    #     target_position = wrist_position + np.array([0.1, 0, 0]) + rotation_matrix.dot(forward_vector)
    #     self.set_camera_pose(camera_point=wrist_position, target_point=target_position)

    def set_camera_pose(self, camera_point, target_point):
        delta_point = np.array(target_point) - np.array(camera_point)
        distance = np.linalg.norm(delta_point)
        dx, dy = delta_point[:2]
        yaw = np.math.atan2(dy, dx) - np.pi/2 # TODO: hack
        dx, dy, dz = delta_point
        pitch = np.math.atan2(dz, np.sqrt(dx ** 2 + dy ** 2))
        p.resetDebugVisualizerCamera(distance, math.degrees(yaw), math.degrees(pitch),
                                    target_point, physicsClientId=self.client)
    def get_wrist_camera_image(self):
        # Get end-effector position and orientation (wrist camera)
        link_state = p.getLinkState(self.robot, 9)  # Link 9 corresponds to the end-effector in Panda URDF
        ee_pos = link_state[4]  # End-effector position
        ee_ori = link_state[5]  # End-effector orientation in quaternion
        print("End orientation (quaternion): ", ee_ori)
        # Convert quaternion to Euler angles for the camera orientation
        # ee_ori_euler = p.getEulerFromQuaternion(ee_ori)
        # print("End orientation (Euler): ", ee_ori_euler)
        rotation_angle_deg = 90  # Rotate the camera by 90 degrees
        angle_rad = np.deg2rad(rotation_angle_deg)
        rot_matrix_90 = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                 0,                 1]
        ])
        
        
        offset = np.array([0.05, 0, 0.05])
        # Set up the camera parameters
        camera_eye = ee_pos + offset  # Camera position (at the wrist)
        # Adjust the camera target to look slightly forward from the end-effector
        # By default, look down the Z-axis of the end-effector
        forward_vector = np.array([0, 0, 1])
        rotation_matrix = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
        rotation_matrix = rot_matrix_90.dot(rotation_matrix)
        offset_target = np.array([0.1, 0, 0])
        # camera_target = ee_pos + np.array([0, 0, -1]) # Camera target (default: look down the Z-axis of the end-effector)
        camera_target = camera_eye + rotation_matrix.dot(forward_vector) + offset_target
        # print("Camera target: ", camera_target)
        # The "up" vector for the camera (usually aligned with the world z-axis)
        camera_up_vector = rotation_matrix.dot([0, 1, 0])
        # camera_up_vector = [0, 1, 0]

        # Compute view matrix
        print(f"Camera eye: {camera_eye}, Camera target: {camera_target}, Camera up: {camera_up_vector}")
        view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up_vector)

        # print(f"Camera eye: {camera_eye}, Camera target: {camera_target}, Camera up: {camera_up_vector}")
        self.camera_extrinsics = view_matrix
        # Set projection parameters
        near = 0.1  # Near clipping plane
        far = 1.0  # Far clipping plane
        fov = 60  # Field of view
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,         # Field of view
            aspect=1.0,     # Aspect ratio
            nearVal=0.1,    # Near clipping plane
            farVal=1.0,     # Far clipping plane
        )
        self.projecion_matrix = projection_matrix
        width, height = 224, 224  # Image resolution

        fov_rad = np.radians(fov)
        f_x = (width / 2) / np.tan(fov_rad / 2)
        f_y = (height / 2) / np.tan(fov_rad / 2)
        c_x = width / 2 
        c_y = height / 2
        intrinsic_matrix = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0,  0,  1]
        ])
        self.camera_intrinsics = intrinsic_matrix
        
        img_data = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # You can use p.ER_BULLET_HARDWARE_OPENGL for better rendering
        )

        # Extract RGB and depth data
        rgb_img = np.reshape(img_data[2], (height, width, 4))[:, :, :3]  # RGB image
        depth_buffer = np.reshape(img_data[3], (height, width))  # Depth buffer

        # Normalize depth buffer to real-world values

        z = depth_buffer * 2.0 - 1.0  # Rescale to range [-1, 1]
        depth_img = 2.0 * near * far / (far + near - z * (far - near))
        
        return rgb_img, depth_buffer

    def depth_to_world(self, depth_img):
        height, width = depth_img.shape
        width, height = depth_img.shape[1], depth_img.shape[0]
        proj_matrix = np.asarray(self.projecion_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.camera_extrinsics).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth_img.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1
        # print("pixels: ", pixels)
        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points
    
    
    def get_camera_intrinsics(self):
        return self.camera_intrinsics

    def get_camera_extrinsics(self):
        return self.camera_extrinsics
    
    def get_projecion_matrix(self):
        return self.projecion_matrix

    def step(self):
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.client)
            # time.sleep(1 / 1000)

    def disconnect(self):
        return p.disconnect()

    def run_test(self, test_length=2000):
        # self.setup_env()
        import random
        self.Videosave_start(video_name="test1")
        ee_pos, ee_ori = self.getEndEffectorPose()
        mean = 0
        std_dev = 0.1

        # Generate Gaussian noise
        rgb_bn, depth_bn = self.get_wrist_camera_image()
        noisy_img = rgb_bn
        num_interactions = 10
        for i in range(num_interactions):
            
            points2 = [(88, 108), (76, 116), (76, 120)]
            points_array = np.array(points2, dtype=np.float32)
            noise = np.random.normal(mean, std_dev, points_array.shape).astype(np.float32)
            noise = noise / 224
            noisy_points_array = points_array + noise
            points2 = [tuple(point) for point in noisy_points_array]
            print("points2: ", points2)
            std_dev += 0.05
        self.Videosave_end()
        print("Test environment")

    
    def Videosave_start(self, video_name):
        
        path = '/home/gentlebear/Mres/dinobot/dinobot/env/' + video_name + '.mp4'
        self.log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, 
            fileName=path, 
            physicsClientId=self.client
        )
        if self.log_id is not None:
            print(f"Started video logging")
        else:
            print("Failed to start video logging.")

    
    def Videosave_end(self):

        if self.log_id is not None:
            p.stopStateLogging(self.log_id)
            print(f"Stopped video logging")
            self.log_id = None

# Example usage:
if __name__ == "__main__":
    env = SimpleEnv(Test_env=True)
    env.run_test()
