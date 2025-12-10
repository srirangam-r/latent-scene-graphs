import pyrealsense2 as rs
import numpy as np
import cv2
import time
from realsense_utils import project_rgbd_to_pointcloud

def main():
    # 1. Setup RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device and enable streams
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    try:
        pipeline_profile = config.resolve(pipeline_wrapper)
    except RuntimeError as e:
        print(f"Error: No RealSense device connected: {e}")
        return

    # Enable aligned depth and color
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")

    # Create alignment object
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Define filters
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    try:
        print("Warming up camera...")
        for _ in range(30):
            pipeline.wait_for_frames()

        print("Capturing aligned frame...")
        frames = pipeline.wait_for_frames()
        
        # Align first
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            print("Error: Could not retrieve frames")
            return

        # Apply filters to aligned depth
        aligned_depth_frame = spatial.process(aligned_depth_frame)
        aligned_depth_frame = temporal.process(aligned_depth_frame)
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)

        # Get Intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        intrinsics_dict = {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy
        }

        # Convert to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert BGR to RGB for saving/processing
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Save data
        filename = 'realsense_data.npz'
        np.savez(filename, 
                 color=color_image_rgb, 
                 depth=depth_image, 
                 intrinsics=intrinsics_dict,
                 depth_scale=depth_scale)
        print(f"Saved RGB-D data to {filename}")

        # Visualize using the utility function
        print("Visualizing captured data...")
        project_rgbd_to_pointcloud(color_image_rgb, depth_image, intrinsics_dict, depth_scale, depth_trunc=1.0, visualize=True)

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
