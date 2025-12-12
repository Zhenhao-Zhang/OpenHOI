import numpy as np

def normalize_point_cloud_to_2048(points):

    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    assert points.ndim == 2 and points.shape[1] == 3, "error"

    N = points.shape[0]
    target_num = 2048

    if N == target_num:
        return points.copy()
    elif N > target_num:
        idx = np.random.choice(N, target_num, replace=False)
    else:
        idx = np.random.choice(N, target_num, replace=True)
    
    return points[idx, :]


if __name__ == "__main__":
    
    input_point_cloud = np.random.rand(5000, 3) # input your pointcloud 
    
    
    output_point_cloud = normalize_point_cloud_to_2048(input_point_cloud)
    
    print("input_shape:", input_point_cloud.shape)
    print("output_shape:", output_point_cloud.shape)  