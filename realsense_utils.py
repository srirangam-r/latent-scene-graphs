import open3d as o3d
import numpy as np
import cv2

def project_rgbd_to_pointcloud(color_image, depth_image, intrinsics_dict, depth_scale=0.001, depth_trunc=1.0, visualize=False):
    """
    Projects RGB and Depth images to a 3D point cloud.

    Args:
        color_image (numpy.ndarray): RGB image (H, W, 3).
        depth_image (numpy.ndarray): Depth image (H, W).
        intrinsics_dict (dict): Dictionary containing 'width', 'height', 'fx', 'fy', 'ppx', 'ppy'.
        depth_scale (float): Scale factor to convert depth values to meters. Default is 0.001 (1mm).
        depth_trunc (float): Depth values larger than this will be truncated (in meters). Default is 1.0.
        visualize (bool): Whether to visualize the point cloud using Open3D.

    Returns:
        open3d.geometry.PointCloud: The generated point cloud.
    """
    
    # Create Open3D intrinsics
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics_dict['width'],
        height=intrinsics_dict['height'],
        fx=intrinsics_dict['fx'],
        fy=intrinsics_dict['fy'],
        cx=intrinsics_dict['ppx'],
        cy=intrinsics_dict['ppy']
    )

    o3d_color = o3d.geometry.Image(color_image)
    o3d_depth = o3d.geometry.Image(depth_image)

    # Create RGBD Image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, 
        o3d_depth, 
        depth_scale=1.0/depth_scale, 
        depth_trunc=depth_trunc, 
        convert_rgb_to_intensity=False
    )

    # Project to Point Cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        o3d_intrinsics
    )

    if visualize:
        print(f"Visualizing point cloud with {len(pcd.points)} points...")
        # Add a coordinate frame to see origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        # Visualize
        o3d.visualization.draw_geometries([pcd, coord_frame])

    return pcd

def create_visualization_geometries(corners_front, normal, extrusion_depth=0.45, cuboid_color=[0, 1, 0]):
    """
    Generates the Pink Grid and Cuboid geometries from the rectangle corners and normal.
    Args:
        corners_front: (4x3) array of front face corners
        normal: (3,) normal vector
        extrusion_depth: Depth of cuboid extrusion (default 0.45m)
        cuboid_color: RGB color for cuboid [R, G, B], default green [0, 1, 0]
    """
    # Ensure normal points towards camera (approx view direction is -center)
    center = np.mean(corners_front, axis=0)
    view_dir = -center
    view_dist = np.linalg.norm(view_dir)
    if view_dist > 0:
        view_dir = view_dir / view_dist
    
    if np.dot(normal, view_dir) < 0:
        normal = -normal
        
    corners_back = corners_front - normal * extrusion_depth

    # Combined vertices: 0-3 (front), 4-7 (back)
    vertices = np.vstack((corners_front, corners_back))
    
    # Lines for cuboid
    cuboid_lines_indices = [
        [0, 1], [1, 2], [2, 3], [3, 0], # Front
        [4, 5], [5, 6], [6, 7], [7, 4], # Back
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connecting
    ]
    
    cuboid_lines = o3d.geometry.LineSet()
    cuboid_lines.points = o3d.utility.Vector3dVector(vertices)
    cuboid_lines.lines = o3d.utility.Vector2iVector(cuboid_lines_indices)
    cuboid_lines.colors = o3d.utility.Vector3dVector([cuboid_color for _ in range(len(cuboid_lines_indices))])
    
    # --- 2. Pink Rectangle (Front Face) - Grid for Transparency ---
    # We will interpolate points along the edges to create a grid
    grid_lines = []
    grid_points = []
    
    # Basis vectors for the rectangle sides
    v1 = corners_front[1] - corners_front[0]
    v2 = corners_front[3] - corners_front[0]
    origin = corners_front[0]
    
    steps = 10 # 10x10 grid
    
    # Add grid points and lines
    idx = 0
    for i in range(steps + 1):
        u = i / steps
        # Vertical lines (along v2)
        p_start = origin + u * v1
        p_end = p_start + v2
        grid_points.append(p_start)
        grid_points.append(p_end)
        grid_lines.append([idx, idx+1])
        idx += 2
        
        # Horizontal lines (along v1)
        p_start = origin + u * v2
        p_end = p_start + v1
        grid_points.append(p_start)
        grid_points.append(p_end)
        grid_lines.append([idx, idx+1])
        idx += 2
        
    pink_grid = o3d.geometry.LineSet()
    pink_grid.points = o3d.utility.Vector3dVector(grid_points)
    pink_grid.lines = o3d.utility.Vector2iVector(grid_lines)
    pink_grid.colors = o3d.utility.Vector3dVector([[1, 0, 1] for _ in range(len(grid_lines))]) # Pink

    return [pink_grid, cuboid_lines]

def fit_plane_rectangle(pcd, priority_normal=None):
    """
    Fits a plane to the point cloud and finds the rectangular extents.
    Args:
        pcd: Open3D PointCloud
        priority_normal: Optional (3,) numpy array. If provided, checks RANSAC result against this.
                         If deviation is large, forces a fit with this normal.
    Returns:
        plane_model: [a, b, c, d]
        visualization_geoms: [pink_grid, cuboid_lines]
        params: {'corners': np.array (4x3), 'normal': np.array (3,)}
    """
    if len(pcd.points) < 3:
        return None, None, None

    # 1. Fit plane using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    ransac_normal = np.array([a, b, c])
    ransac_normal = ransac_normal / np.linalg.norm(ransac_normal)

    # 2. Check against priority normal
    use_fixed_normal = False
    if priority_normal is not None:
        # Check simple dot product (ignoring flip for now, assuming similar orientation)
        # But plane normal could be flipped 180.
        # dot > 0 means same dir, < 0 means opposite.
        # We want to check alignment: abs(dot)
        dot_p = np.dot(ransac_normal, priority_normal)
        
        # If alignment is poor (< 0.9, approx 25 deg)
        if abs(dot_p) < 0.9:
            # Fallback to priority normal
            # print(f"Normal deviation detected ({abs(dot_p):.3f}). Forcing priority normal.")
            use_fixed_normal = True
            final_normal = priority_normal
        else:
            # Good alignment, keep RANSAC but ensure direction matches priority
            if dot_p < 0:
                ransac_normal = -ransac_normal
                d = -d 
                plane_model = [ransac_normal[0], ransac_normal[1], ransac_normal[2], d]
            final_normal = ransac_normal
    else:
        final_normal = ransac_normal

    if use_fixed_normal:
        # Fit d for fixed normal
        # minimize sum( (n.p + d)^2 ) -> d = -mean(n.p)
        # Robust: d = -median(n.p)
        points = np.asarray(pcd.points)
        projections = np.dot(points, final_normal)
        d_best = -np.median(projections)
        plane_model = [final_normal[0], final_normal[1], final_normal[2], d_best]

    # Extract updated model
    [a, b, c, d] = plane_model
    
    # 3. Project points to plane (Standard Logic)
    # Z axis
    z_axis = np.array([0, 0, 1])
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    if np.abs(np.dot(normal, z_axis)) > 0.99:
        R = np.eye(3)
    else:
        v = np.cross(normal, z_axis)
        c_val = np.dot(normal, z_axis)
        s_val = np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]], 
                      [v[2], 0, -v[0]], 
                      [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c_val) / (s_val ** 2))

    # Transform points
    points = np.asarray(pcd.points)
    points_rot = points @ R.T
    
    # Now aligned to Z, take X and Y coordinates
    # For better fit, we should do PCA on the x,y points to find OBB in 2D
    points_2d = points_rot[:, :2]
    
    # PCA for 2D orientation
    mean_2d = np.mean(points_2d, axis=0)
    centered_2d = points_2d - mean_2d
    cov = np.cov(centered_2d.T)
    evals, evecs = np.linalg.eig(cov)
    
    # Sort simple
    idx = evals.argsort()[::-1]
    evecs = evecs[:, idx]
    
    # Rotate 2D points to align with PCA axes
    points_pca = centered_2d @ evecs
    
    # Calculate box corners
    min_x, min_y = np.min(points_pca, axis=0)
    max_x, max_y = np.max(points_pca, axis=0)
    
    # 4 corners in PCA space
    corners_pca = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
    
    # Transform back to 2D (original frame, but rotated)
    corners_2d = corners_pca @ evecs.T + mean_2d
    
    # Add Z coordinate (mean Z of rotated points)
    mean_z = np.mean(points_rot[:, 2])
    corners_rot = np.column_stack((corners_2d, np.full(4, mean_z)))
    
    # Transform back to 3D world space
    corners_world = corners_rot @ R
    
    # Calculate center of rectangle
    center = np.mean(corners_world, axis=0)
    
    # Move towards camera (origin is at 0,0,0) instead of along normal
    # This ensures it visually pops "forward" regardless of plane orientation
    view_dir = -center
    view_dist = np.linalg.norm(view_dir)
    if view_dist > 0:
        view_dir = view_dir / view_dist
        
    # Offset by 1cm towards camera
    offset = 0.01
    corners_front = corners_world + view_dir * offset
    
    params = {
        'corners': corners_front,
        'normal': normal
    }
    
    geoms = create_visualization_geometries(corners_front, normal)
    
    return plane_model, geoms, params
def create_back_plane_geometry(corners):
    """
    Creates a purple grid plane for the back face visualization.
    Args:
        corners: (4x3) array of back face corners
    Returns:
        LineSet representing the purple back plane grid
    """
    grid_lines = []
    grid_points = []
    
    # Basis vectors for the rectangle sides
    v1 = corners[1] - corners[0]
    v2 = corners[3] - corners[0]
    origin = corners[0]
    
    steps = 10  # 10x10 grid
    
    # Add grid points and lines
    idx = 0
    for i in range(steps + 1):
        u = i / steps
        # Vertical lines (along v2)
        p_start = origin + u * v1
        p_end = p_start + v2
        grid_points.append(p_start)
        grid_points.append(p_end)
        grid_lines.append([idx, idx+1])
        idx += 2
        
        # Horizontal lines (along v1)
        p_start = origin + u * v2
        p_end = p_start + v1
        grid_points.append(p_start)
        grid_points.append(p_end)
        grid_lines.append([idx, idx+1])
        idx += 2
        
    purple_grid = o3d.geometry.LineSet()
    purple_grid.points = o3d.utility.Vector3dVector(grid_points)
    purple_grid.lines = o3d.utility.Vector2iVector(grid_lines)
    purple_grid.colors = o3d.utility.Vector3dVector([[0.5, 0, 0.5] for _ in range(len(grid_lines))]) # Purple
    
    return purple_grid
