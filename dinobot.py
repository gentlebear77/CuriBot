"""
In this script, we demonstrate how to use DINOBot to do one-shot imitation learning.
You first need to install the following repo and its requirements: https://github.com/ShirAmir/dino-vit-features.
You can then run this file inside that repo.
There are a few setup-dependent functions you need to implement, like getting an RGBD observation from the camera
or moving the robot, that you will find on top of this file.
"""

import sys
sys.path.append('franka-pybullet/src')
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torchvision import transforms,utils
from PIL import Image
import torchvision.transforms as T
import warnings 
import glob
import time
import cv2
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
import visualizer.visualizer.pointcloud as visualizer
from scipy.spatial.transform import Rotation
warnings.filterwarnings("ignore")

#Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features
from correspondences import find_correspondences, draw_correspondences

from dinobot.env.base import SimpleEnv

stepsize = 1 / 500
robot_origin = SimpleEnv()
# robot_origin.reset()

#Hyperparameters for DINO correspondences extraction
num_pairs = 6
load_size = 224 
layer = 8
facet = 'key' 
bin=True 
thresh=0.6
model_type='dino_vits8' 
stride=4 
#Deployment hyperparameters    
ERR_THRESHOLD = 0.5 #A generic error between the two sets of points

class Dinobot:
    def __init__(self, robot):
        self.video = False
        video_name = 'noise_test'
        with open('data_collection/demo/robot_demo_delta_position.pkl', 'rb') as f:
            self.demo_data = pickle.load(f)
        
        #Get rgbd from wrist camera.
        rgb_bn, depth_bn = origin_camera_get_rgbd(demo_data=demo_data)
        intrinsics = robot_origin.get_camera_intrinsics()

        robot_origin.disconnect()
        robot = SimpleEnv(Test_env=True)
        

    def replay(self):
        pass

#Here are the functions you need to create based on your setup.
def camera_get_rgbd():
    """
    Outputs a tuple (rgb, depth) taken from a wrist camera.
    The two observations should have the same dimension.
    """
    rgb_bn, depth_bn = robot.get_wrist_camera_image()
    
    if rgb_bn.shape[:2] != depth_bn.shape[:2]:
        # Resize the depth image to match the dimensions of the RGB image
        depth_bn_resized = cv2.resize(depth_bn, (rgb_bn.shape[1], rgb_bn.shape[0]))
    else:
        depth_bn_resized = depth_bn

    cv2.imwrite('rgb_bn.png', rgb_bn)
    return (rgb_bn, depth_bn_resized)

def origin_camera_get_rgbd(demo_data):
    """
    Outputs a tuple (rgb, depth) taken from a wrist camera.
    The two observations should have the same dimension.
    """
    # position = demo_data['joint_position'][0]
    # gripper_state = demo_data['gripper_states'][0]
    # # Set the recorded joint states for this timestep
    # robot_origin.setTargetPositions(position)

    # # Control the gripper based on recorded gripper state
    # robot_origin.controlGripper(gripper=gripper_state)

    # Step the simulation to the next timestep
    # robot_origin.step()

    rgb_bn, depth_bn = robot_origin.get_wrist_camera_image()

    if rgb_bn.shape[:2] != depth_bn.shape[:2]:
        # Resize the depth image to match the dimensions of the RGB image
        depth_bn_resized = cv2.resize(depth_bn, (rgb_bn.shape[1], rgb_bn.shape[0]))
    else:
        depth_bn_resized = depth_bn

    camera_view_matrix = robot_origin.get_camera_extrinsics()

    cv2.imwrite('rgb_bn_origin.png', rgb_bn)
    return (rgb_bn, depth_bn_resized)
    
def project_to_3d(points, depth, intrinsics):
    """
    Inputs: points: list of [x,y] pixel coordinates, 
            depth (H,W,1) observations from camera.
            intrinsics: intrinsics of the camera, used to 
            project pixels to 3D space.
    Outputs: point_with_depth: list of [x,y,z] coordinates.
    
    Projects the selected pixels to 3D space using intrinsics and
    depth value. Based on your setup the implementation may vary,
    but here you can find a simple example or the explicit formula:
    https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.create_point_cloud_from_rgbd_image.html.
    """
    view_matrix = robot.get_camera_extrinsics()
    
    # points_world = robot.depth_to_world(depth, intrinsics, camera_pose_world)
    proj_matrix = np.asarray(robot.get_projecion_matrix()).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
    if isinstance(intrinsics, np.ndarray):
        width, height = depth.shape[1], depth.shape[0]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        # print(f'width: {width}, height: {height}, fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}')
        intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    depth_values = np.array([depth[int(y), int(x)] for x, y in points])
    points_world = robot.depth_to_world(depth)

    normalized_z = 2 * depth_values - 1
    normalized_x = 2 * np.array([x for x, y in points]) / width - 1
    normalized_y = 2 * (1 - np.array([y for x, y in points]) / height) - 1
    # print(f'normalized_x: {normalized_x}, normalized_y: {normalized_y}, normalized_z: {normalized_z}')
    pixels = np.stack([normalized_x, normalized_y, normalized_z, np.ones_like(normalized_z)], axis=1)
    target_points_world = np.matmul(tran_pix_world, pixels.T).T
    target_points_world /= target_points_world[:, 3:4]
    target_points_world = target_points_world[:, :3]
    print(f'target_points_world: {target_points_world}')

    # print(f'point_with_depth: {point_with_depth}')
    # print(f'point_with_depth: {point_with_depth_world}')
    points_np = np.array(target_points_world)
    # print(f'point_with_depth: {points_np}')
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_np)
    visualizer.save_visualization_to_file(points_world, f'pointcloud{points[0]}.html')
    visualizer.save_visualization_to_file(target_points_world, f'pointcloud{points[0]}_dino_point.html')

    # o3d.visualization.draw_geometries([pcd])

    return target_points_world


def robot_move(t_meters, R, duration=3):
    """
    Inputs: t_meters: (x,y,z) translation in end-effector frame
            R: (3x3) array - rotation matrix in end-effector frame
    
    Moves and rotates the robot according to the input translation and rotation.
    """

    # Convert rotation matrix to quaternion
    delta_rotation = Rotation.from_matrix(R)
    # PyBullet expect quaternions as [w, x, y, z]
    current_pos, current_ori = robot.getEndEffectorPose()
    print(f'current_pos: {current_pos}')
    print(f't_meters: {t_meters}')
    current_rotation = Rotation.from_quat([current_ori[1], current_ori[2], current_ori[3], current_ori[0]])
    # t_meters = [0, 0.2, 0]
    target_pos = [
        current_pos[0] + t_meters[1],
        current_pos[1] + t_meters[0],
        current_pos[2] + t_meters[2]
    ]
    
    target_rotation = current_rotation * delta_rotation
    # flip_z_rotation = Rotation.from_euler('z', 180, degrees=True)
    # target_rotation = current_rotation
    # target_rotation = target_rotation * flip_z_rotation
    
    print(f'target_rotation: {(target_rotation.as_euler("xyz", degrees=True))}')
    target_quaternion = target_rotation.as_quat()

    target_quaternion_wxyz = [target_quaternion[3], target_quaternion[0], target_quaternion[1], target_quaternion[2]]

    target_joint_positions = robot.InverseKinematics(target_pos, target_quaternion_wxyz)
    
    # print(f'target_joint_positions: {target_joint_positions}')    
    robot.setTargetPositions(target_joint_positions)

    num_steps = int(duration / stepsize)

    robot.simulate_step()
    print("posterior position", robot.getEndEffectorPose())
    
def replay_demo(demo_data):
    """
    Inputs: demo: list of velocities that can then be executed by the end-effector.
    Replays a demonstration by moving the end-effector given recorded velocities.
    """

    # Get the recorded data from the pickle file
    timestamps = demo_data['timestamps']
    gripper_states = demo_data['gripper_states']
    delta_positions = demo_data['delta_end_effector_position']
    delta_orientations = demo_data['delta_end_effector_orientation']
    gripper_state = False
    # Replay the demo
    for timestep, gripper_open, delta_position, delta_orientation in zip(timestamps, gripper_states, delta_positions, delta_orientations):
        # print("Simulation time: {:.3f}".format(timestep))
        # Set the recorded joint states for this timestep
        robot.setDeltaEndControl(delta_position, delta_orientation)
        # _, robot_velocities = robot.getJointStates()
        # error = np.linalg.norm(np.array(joint_positions) - np.array(previous_joint_positions + delta_position))
        # if error:
        #     print(error)
        if gripper_open != gripper_state:
            gripper_state = gripper_open
            robot.control_gripper(gripper=gripper_open)

        robot.simulate_step()

    # raise NotImplementedError
    

def find_transformation(X, Y):
    """
    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    Find transformation given two sets of correspondences between 3D points.
    """
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    print(f'cX: {cX}, cY: {cY}')
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t

def compute_error(points1, points2):
    return np.linalg.norm(np.array(points1) - np.array(points2))


if __name__ == "__main__":
 
    # RECORD DEMO:
    # Move the end-effector to the bottleneck pose and store observation.
    
    #Record demonstration.
    # demo_vels = record_demo()
    video = False
    video_name = 'noise_test'
    with open('data_collection/demo/robot_demo_delta_position.pkl', 'rb') as f:
        demo_data = pickle.load(f)
    
    #Get rgbd from wrist camera.
    rgb_bn, depth_bn = origin_camera_get_rgbd(demo_data=demo_data)
    intrinsics = robot_origin.get_camera_intrinsics()

    robot_origin.disconnect()
    robot = SimpleEnv(Test_env=True)
    
    # robot.reset()
    
    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    while 1:
        error = 100000
        while error > ERR_THRESHOLD:
            #Collect observations at the current pose.
            rgb_live, depth_live = camera_get_rgbd()

            #Compute pixel correspondences between new observation and bottleneck observation.

            with torch.no_grad():
                
            #This function from an external library takes image paths as input. Therefore, store the paths of the
            #observations and then pass those
                # print(f'rgb_bn: {rgb_bn.shape, type(rgb_bn)}, rgb_live: {rgb_live.shape}')
                rgb_bn_dic = 'rgb_bn_origin.png'
                rgb_live_dic = 'rgb_bn.png'
                points1, points2, image1_pil, image2_pil = find_correspondences(rgb_bn_dic, rgb_live_dic, num_pairs, load_size, layer,
                                                                                       facet, bin, thresh, model_type, stride)
                
            #    Draw correspondences
                fg1, fg2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
                fg1.savefig("fg1.pdf", dpi=224, bbox_inches='tight')
                fg2.savefig("fg2.pdf", dpi=224, bbox_inches='tight')
            #     plt.close()
            #     # plt.show()

            #Given the pixel coordinates of the correspondences, and their depth values,
            #project the points to 3D space.

            # points1 = [(76, 116)]
            # # points2 = [(44, 84)]

            # points1 = [(84, 120), (84, 104), (76, 108), (88, 108), (76, 116), (72, 116)]
            # points2 = [(80, 88), (84, 56), (72, 68), (88, 64), (72, 84), (64, 80)]
            print(f'points1: {points1}, points2: {points2}')
            common_points = set(points1) & set(points2)
            # Remove common tuples from both lists
            points1 = [point for point in points1 if point not in common_points]
            points2 = [point for point in points2 if point not in common_points]

            print(f'points1: {points1}, points2: {points2}')
            points1 = project_to_3d(points1, depth_bn, intrinsics)
            points2 = project_to_3d(points2, depth_live, intrinsics)
            # print(f'points1_after_project: {points1}\n, points2_after_project: {points2}')
            #Find rigid translation and rotation that aligns the points by minimising error, using SVD.
            R, t = find_transformation(points1, points2)
            print(f'R: {R}, t: {t}')
           
            robot_move(t,R)
            error = compute_error(points1, points2)
            print(f'Error: {error}')

        #Once error is small enough, replay demo.
        replay_demo(demo_data)

        print("Demo replayed.")
        break