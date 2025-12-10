import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import shutil
from datetime import datetime

def main():
    # 1. Setup Directories
    output_dir = "video_data"
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(rgb_dir)
    os.makedirs(depth_dir)
    
    # 2. Setup RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams (VGA @ 30fps is standard and good for SAM2)
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
        # Save Intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        intrinsics_dict = {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy,
            'depth_scale': depth_scale
        }
        
        with open(os.path.join(output_dir, "intrinsics.json"), "w") as f:
            json.dump(intrinsics_dict, f, indent=4)
            
        print("Recording... Press 'q' to stop.")
        frame_idx = 0
        
        while True:
            frames = pipeline.wait_for_frames()
            
            # Align
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
                
            # Apply filters
            aligned_depth_frame = spatial.process(aligned_depth_frame)
            aligned_depth_frame = temporal.process(aligned_depth_frame)
            aligned_depth_frame = hole_filling.process(aligned_depth_frame)
            
            # Convert to numpy
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Save frames
            # RGB: JPEG
            cv2.imwrite(os.path.join(rgb_dir, f"{frame_idx:06d}.jpg"), color_image)
            
            # Depth: PNG (16-bit)
            # cv2.imwrite saves 16-bit PNG correctly if input is uint16
            cv2.imwrite(os.path.join(depth_dir, f"{frame_idx:06d}.png"), depth_image)
            
            # Visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            
            cv2.imshow('Recording RealSense', images)
            key = cv2.waitKey(1)
            
            if key & 0xFF == ord('q'):
                break
                
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Recorded {frame_idx} frames...")
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Finished. Saved {frame_idx} frames to '{output_dir}'.")

if __name__ == "__main__":
    main()
