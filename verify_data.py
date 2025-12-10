import os
import cv2
import json
import numpy as np
import open3d as o3d
from realsense_utils import project_rgbd_to_pointcloud

def main():
    video_dir = "video_data"
    rgb_dir = os.path.join(video_dir, "rgb")
    depth_dir = os.path.join(video_dir, "depth")
    intrinsics_path = os.path.join(video_dir, "intrinsics.json")
    
    if not os.path.exists(video_dir):
        print(f"Error: {video_dir} not found.")
        return

    # Load Intrinsics
    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    print(f"Loaded intrinsics: {intrinsics_dict}")
    
    # Get first frame
    frame_names = sorted([p for p in os.listdir(rgb_dir) if p.endswith((".jpg", ".jpeg"))])
    if not frame_names:
        print("No RGB frames found.")
        return
        
    frame_name = frame_names[0]
    rgb_path = os.path.join(rgb_dir, frame_name)
    depth_name = frame_name.replace(".jpg", ".png").replace(".jpeg", ".png")
    depth_path = os.path.join(depth_dir, depth_name)
    
    print(f"Processing Frame: {frame_name}")
    print(f"RGB Path: {rgb_path}")
    print(f"Depth Path: {depth_path}")
    
    if not os.path.exists(depth_path):
        print("Error: Corresponding depth file not found.")
        return

    # Load Images
    color_image = cv2.imread(rgb_path)
    if color_image is None:
        print("Error: Failed to load RGB image.")
        return
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        print("Error: Failed to load Depth image.")
        return
        
    print(f"RGB Shape: {color_image.shape}")
    print(f"Depth Shape: {depth_image.shape}")
    print(f"Depth Min/Max: {depth_image.min()}/{depth_image.max()}")
    
    # Project and Visualize
    print("Projecting to Point Cloud...")
    pcd = project_rgbd_to_pointcloud(
        color_image, 
        depth_image, 
        intrinsics_dict, 
        depth_scale=intrinsics_dict.get('depth_scale', 0.001),
        depth_trunc=3.0, # View up to 3m
        visualize=True
    )
    
if __name__ == "__main__":
    main()
