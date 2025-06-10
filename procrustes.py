import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Example points (replace these with the actual points from your figures)
# Source points (points in the left figure)
source_points = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 5],
    [5, 6]
])

# Target points (points in the right figure)
target_points = np.array([
    [2, 2],
    [3, 3],
    [4, 2],
    [5, 6],
    [6, 7]
])

# Convert the points to Open3D PointCloud objects
source_pcd = o3d.geometry.PointCloud()
target_pcd = o3d.geometry.PointCloud()

source_pcd.points = o3d.utility.Vector3dVector(source_points)
target_pcd.points = o3d.utility.Vector3dVector(target_points)

# Apply ICP to align the source points to the target points
threshold = 0.02  # Maximum distance threshold for ICP
trans_init = np.eye(4)  # Initial transformation (identity matrix)

# Perform the ICP alignment
reg_icp = o3d.registration.registration_icp(
    source_pcd, target_pcd, threshold, trans_init,
    o3d.registration.TransformationEstimationPointToPoint())

# Get the resulting transformation matrix and aligned source points
transformation = reg_icp.transformation
aligned_source_points = np.asarray(source_pcd.points)

# Apply the transformation to the source points
source_pcd.transform(transformation)

# Visualize the result
# Color the source and target points differently
source_pcd.paint_uniform_color([1, 0, 0])  # Red for source
target_pcd.paint_uniform_color([0, 1, 0])  # Green for target

# Plot the source and target points after alignment
o3d.visualization.draw_geometries([source_pcd, target_pcd])

print("ICP Transformation Matrix:")
print(transformation)
