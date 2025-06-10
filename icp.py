import open3d as o3d
import copy
import numpy as np

# Initialize functions
def draw_registration_result(source, target, transformation):
    """
    param: source - source point cloud
    param: target - target point cloud
    param: transformation - 4 X 4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], zoom=0.4459, front=[0.9288, -0.2951, -0.2242], lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])

def find_nearest_neighbors(source_pc, target_pc, nearest_neigh_num):
    # Find the closest neighbor for each anchor point through KDTree
    point_cloud_tree = o3d.geometry.KDTreeFlann(source_pc)
    # Find nearest target_point neighbor index
    points_arr = []
    for point in target_pc.points:
        [_, idx, _] = point_cloud_tree.search_knn_vector_3d(point, nearest_neigh_num)
        points_arr.append(source_pc.points[idx[0]])
    return np.asarray(points_arr)


def icp(source, target):
    source.paint_uniform_color([0.5, 0.5, 0.5])
    target.paint_uniform_color([0, 0, 1])
    #source_points = np.asarray(source.points) # source_points is len()=198835x3 <--> 198835 points that have (x,y,z) val
    target_points = np.asarray(points1)
    # Since there are more source_points than there are target_points, we know there is not
    # a perfect one-to-one correspondence match. Sometimes, many points will match to one point,
    # and other times, some points may not match at all.

    # transform_matrix = np.asarray([[0.862, 0.011, -0.507, 0.5], [-0.139, 0.967, -0.215, 0.7], [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # source = source.transform(transform_matrix)

    # While loop variables
    curr_iteration = 0
    cost_change_threshold = 0.0001
    curr_cost = 1000
    prev_cost = 10000

    while (True):
        # 1. Find nearest neighbors
        new_source_points = find_nearest_neighbors(source, target, 1)
        print("new_source_points=", new_source_points)
        # 2. Find point cloud centroids and their repositions
        source_centroid = np.mean(new_source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        source_repos = np.zeros_like(new_source_points)
        target_repos = np.zeros_like(target_points)
        source_repos = np.asarray([new_source_points[ind] - source_centroid for ind in range(len(new_source_points))])
        target_repos = np.asarray([target_points[ind] - target_centroid for ind in range(len(target_points))])

        # 3. Find correspondence between source and target point clouds
        cov_mat = target_repos.transpose() @ source_repos

        U, X, Vt = np.linalg.svd(cov_mat)
        R = U @ Vt
        t = target_centroid - R @ source_centroid
        t = np.reshape(t, (1,3))
        curr_cost = np.linalg.norm(target_repos - (R @ source_repos.T).T)
        print("Curr_cost=", curr_cost)
        if ((prev_cost - curr_cost) > cost_change_threshold):
            prev_cost = curr_cost
            transform_matrix = np.hstack((R, t.T))
            transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 0, 1])))
            print("Transform_matrix=", transform_matrix)
            # If cost_change is acceptable, update source with new transformation matrix
            source = source.transform(transform_matrix)
            curr_iteration += 1
        else:
            break
    print("\nIteration=", curr_iteration)
    # Visualize final iteration and print out final variables
    draw_registration_result(source, target, transform_matrix)
    return transform_matrix

### PART A ###


# Provided points
points1 = [[4.34240786e-01, 6.64439671e-02, 3.30471475e-04], 
           [4.07585709e-01, 5.29365279e-02, 3.29124877e-04], 
           [4.07585709e-01, 6.61706599e-02, 3.29124877e-04]]
points2 = [[5.15585001e-01, 1.34590216e-01, 1.68088960e-04], 
           [4.88251033e-01, 1.20626584e-01, 1.67401641e-04], 
           [4.88251033e-01, 1.34029537e-01, 1.67401641e-04]]

# Convert to numpy arrays
points1_np = np.array(points1)
points2_np = np.array(points2)

# Create point clouds from the points
source = o3d.geometry.PointCloud()
target = o3d.geometry.PointCloud()

# Set the points to the point clouds
source.points = o3d.utility.Vector3dVector(points1_np)
target.points = o3d.utility.Vector3dVector(points2_np)


part_a = icp(source, target)

### PART B ###

source = o3d.io.read_point_cloud("kitti_frame1.pcd")
target = o3d.io.read_point_cloud("kitti_frame2.pcd")
print(source.points, target.points)
part_b = icp(source, target)