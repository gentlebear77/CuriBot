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
import random
import open3d as o3d
import matplotlib.pyplot as plt
import visualizer.visualizer.pointcloud as visualizer
from segement import SAMSegmentation

from sklearn.neighbors import KernelDensity
from shapely.geometry import Point, Polygon
from scipy.spatial import KDTree

from scipy.spatial.transform import Rotation
warnings.filterwarnings("ignore")
from correspondences import find_correspondences, draw_correspondences
from cluster import dbscan, get_largest_cluster

#Hyperparameters for DINO correspondences extraction
num_pairs = 8
load_size = 224 
layer = 9
facet = 'key' 
bin=True 
thresh=0.1
model_type='dino_vits8' 
stride=4 

def remove_points_outside_boundary(points, boundary_coords):
    """
    Remove points that are outside the boundary defined by the coordinates.
    """
    boundary_array = np.array(boundary_coords, dtype=np.int32)
    index_mask = np.array([0] * len(points))
    # Filter points that are inside the boundary
    inside_points = []

    for i in range(len(points)):
        point = points[i]
        # Use pointPolygonTest to check if the point is inside the polygon (positive distance means inside)
        distance = cv2.pointPolygonTest(boundary_array, tuple(point), measureDist=True)
        
        # If distance is greater than or equal to 0, the point is inside or on the boundary
        if distance >= 0:
            inside_points.append(point)
            index_mask[i] = 1

    return index_mask
    
def sample_points_in_polygon(polygon, n_samples):
    valid_samples = []
    while len(valid_samples) < n_samples:
        # Randomly sample a point
        point = [random.uniform(polygon.bounds[0], polygon.bounds[2]),  # x-coordinate
                 random.uniform(polygon.bounds[1], polygon.bounds[3])]  # y-coordinate
        
        # Check if the point is inside the polygon
        if polygon.contains(Point(point)):
            valid_samples.append(point)
    return np.array(valid_samples)



if __name__ == "__main__":

    print(torch.cuda.is_available())
    with torch.no_grad():
                
        #This function from an external library takes image paths as input. Therefore, store the paths of the
        #observations and then pass those
        # print(f'rgb_bn: {rgb_bn.shape, type(rgb_bn)}, rgb_live: {rgb_live.shape}')
        # rgb_bn_dic = '/home/gentlebear/Mres/dinobot/realsense_image/color_image_1.png'
        # rgb_live_dic = '/home/gentlebear/Mres/dinobot/realsense_image/color_image.png'
        # rgb_bn_dic = 'rgb_bn_origin.png'
        # rgb_live_dic = 'rgb_bn.png'
        # points1, points2, image1_pil, image2_pil = find_correspondences(rgb_bn_dic, rgb_live_dic, num_pairs, load_size, layer,
        #                                                                         facet, bin, thresh, model_type, stride)
        points1 = [(84, 112), (80, 120), (88, 112), (80, 116), (84, 108)]
        points2 = [(88, 128), (80, 136), (100, 124), (80, 132), (88, 124)]
        common_points = set(points1) & set(points2)
        # Remove common tuples from both lists
        points1 = [point for point in points1 if point not in common_points]
        points2 = [point for point in points2 if point not in common_points]
        print(f'points1: {points1}, points2: {points2}')
        points2 = np.array(points2)
        points1 = np.array(points1)
        cluster_labels = dbscan(points2, eps=10, min_samples=1)
        print(f'cluster_labels: {cluster_labels}')
        points1 = get_largest_cluster(points1, cluster_labels)
        points2 = get_largest_cluster(points2, cluster_labels)
        print(f'points1: {points1}, points2: {points2}')
    #     segementer = SAMSegmentation(image_path = rgb_live_dic)
    #     segementer.process_image()
    #     boundary = segementer.analyze_masks()
    #     boundary_coords = boundary[0]['coordinates'][0]
    #     boundary_coords = [[int(y), int(x)] for x, y in boundary_coords]

    #     points_mask = remove_points_outside_boundary(points2, boundary_coords)
    #     points2 = np.array(points2)[points_mask == 1]
    #     points1 = np.array(points1)[points_mask == 1]
    #     print(f'points1: {points1}, points2: {points2}')
    # #   points1: [[ 84 116] [ 84 108] [ 92 116] [ 84 112]], 
    # #   points2: [[40 88] [44 80] [72 92] [40 84]]
    # # #    Draw correspondences
    # #     fg1, fg2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
    # #     fg1.savefig(f"{rgb_bn_dic}_origin.pdf", dpi=224, bbox_inches='tight')
    # #     fg2.savefig(f"{rgb_live_dic}.pdf", dpi=224, bbox_inches='tight')
    # #     plt.close()
    # centroid1 = np.mean(points1, axis=0)
    # centroid2 = np.mean(points2, axis=0)

    # boundary_coords = np.array(boundary_coords) - centroid2
    # polygon = Polygon(boundary_coords)
    # # Estimate KDE for each set
    # kde1 = KernelDensity(kernel='gaussian', bandwidth=0.1)
    # kde1.fit(points1)
    # kde2 = KernelDensity(kernel='gaussian', bandwidth=0.1)
    # kde2.fit(points2)

    # n_samples = 100
    # sampled_points = sample_points_in_polygon(polygon, n_samples)
    # # print(f'sampled_points: {sampled_points}')
    # # Sample points from the combined distribution
    
    # log_prob1 = kde1.score_samples(sampled_points)
    # log_prob2 = kde2.score_samples(sampled_points)
    # # print(f'log_prob1: {log_prob1}, log_prob2: {log_prob2}')
    # weight1 = 0.5
    # weight2 = 0.5
    # log_prob = weight1 * log_prob1 + weight2 * log_prob2
    # # Trick to avoid numerical underflow
    # log_prob -= np.max(log_prob)  # For numerical stability
    # print(f'log_prob: {log_prob}')
    # prob = np.exp(log_prob)  # Convert log density to probability
    # # Normalize the probabilities so they sum to 1
    # prob /= np.sum(prob)
    # print(f'prob: {prob}')

    # # Sample based on the computed probability distribution
    # sampled_indices = np.random.choice(n_samples, size=3, p=prob)
    # final_sampled_points = sampled_points[sampled_indices]
    # print(f'final_sampled_points: {final_sampled_points}')
