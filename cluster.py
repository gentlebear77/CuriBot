import numpy as np

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def dbscan(points, eps, min_samples):
    """
    A simple DBSCAN algorithm that clusters points based on distance.
    
    :param points: A 2D array of points (each row is a point).
    :param eps: Maximum distance between two points for them to be considered neighbors.
    :param min_samples: Minimum number of points required to form a cluster.
    
    :return: Cluster labels for each point, where -1 indicates noise.
    """
    # Number of points
    n_points = len(points)
    
    # Labels for each point: -1 means unclassified (noise initially)
    labels = -1 * np.ones(n_points)
    
    # Core points, visited points
    visited = np.zeros(n_points, dtype=bool)
    
    def region_query(point_idx):
        """Find all points within eps distance from the point at point_idx."""
        neighbors = []
        for i in range(n_points):
            if euclidean_distance(points[point_idx], points[i]) <= eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(point_idx, neighbors, cluster_id):
        """Expand the cluster by recursively adding all reachable points."""
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend(new_neighbors)
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            i += 1

    cluster_id = 0
    for point_idx in range(n_points):
        if visited[point_idx]:
            continue
        
        visited[point_idx] = True
        neighbors = region_query(point_idx)
        
        if len(neighbors) < min_samples:
            continue  # Mark as noise, no cluster is formed
        
        # Start a new cluster
        expand_cluster(point_idx, neighbors, cluster_id)
        cluster_id += 1
    
    return labels

def get_largest_cluster(points, cluster_labels):
    # Find unique cluster labels (excluding noise labeled as -1)
    unique_labels = np.unique(cluster_labels)
    largest_cluster_id = None
    largest_cluster_size = 0
    
    # Iterate through the unique labels to find the largest cluster (ignoring -1 which is noise)
    for label in unique_labels:
        if label != -1:  # Ignore noise points
            cluster_size = np.sum(cluster_labels == label)
            if cluster_size > largest_cluster_size:
                largest_cluster_size = cluster_size
                largest_cluster_id = label

    # Extract points belonging to the largest cluster
    largest_cluster_points = points[cluster_labels == largest_cluster_id]
    
    return largest_cluster_points

# Example usage:
if __name__ == "__main__":
    # Generate some random data points for clustering
    points = np.random.rand(100, 2)  # 100 points in 2D space

    # Set DBSCAN parameters
    eps = 0.1  # Distance threshold for neighbors
    min_samples = 3  # Minimum number of points to form a cluster
    
    # Apply DBSCAN
    cluster_labels = dbscan(points, eps, min_samples)

    # Output the cluster labels
    print("Cluster labels for each point:", cluster_labels)
