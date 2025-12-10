import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import shutil
from datetime import datetime

def main():
    # -------------------------------------------------------------------------
    # 1. SETUP DIRECTORIES
    # -------------------------------------------------------------------------
    output_dir = "video_data"
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(rgb_dir)
    os.makedirs(depth_dir)
    
    # -------------------------------------------------------------------------
    # 2. SETUP REALSENSE PIPELINE & HARDWARE
    # -------------------------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams (VGA @ 30fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # --- HARDWARE OPTIMIZATION ---
    # Get the depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    
    # Preset 4 = "High Density". Helps significantly with filling gaps.
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 4)
        
    # Maximize Laser Power (improves stability on dark surfaces)
    if depth_sensor.supports(rs.option.laser_power):
        max_laser = depth_sensor.get_option_range(rs.option.laser_power).max
        depth_sensor.set_option(rs.option.laser_power, max_laser)

    # Get depth scale
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")
    
    # Create alignment object (align Depth -> Color)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # -------------------------------------------------------------------------
    # 3. DEFINE FILTERS (Anti-Flicker Tuned)
    # -------------------------------------------------------------------------
    # Decimation: Reduce resolution slightly to reduce noise (Optional, set to 1 to keep original size)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)

    # Spatial Filter: Smooths the geometry
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    
    # Temporal Filter: The heavy lifter for FLICKER reduction
    temporal = rs.temporal_filter()
    # Smooth Alpha: 0.1 is very smooth (relies on past frames). Prevents value jitter.
    temporal.set_option(rs.option.filter_smooth_alpha, 0.1)
    # Smooth Delta: Threshold for change
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    # Persistence (Holes Fill): 3 = Valid in 2/last 8 frames. Prevents black flashing.
    temporal.set_option(rs.option.holes_fill, 3) 
    
    # Hole Filling: Fills remaining small black gaps
    hole_filling = rs.hole_filling_filter()
    # Mode 1 = Fills from left (good for object edges)
    hole_filling.set_option(rs.option.holes_fill, 1)

    # Disparity Transforms (Required for correct Spatial filtering)
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    
    try:
        # Save Intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        intrinsics_dict = {
            'width': intrinsics.width, 'height': intrinsics.height,
            'fx': intrinsics.fx, 'fy': intrinsics.fy,
            'ppx': intrinsics.ppx, 'ppy': intrinsics.ppy,
            'depth_scale': depth_scale
        }
        
        with open(os.path.join(output_dir, "intrinsics.json"), "w") as f:
            json.dump(intrinsics_dict, f, indent=4)
            
        print("Recording... Press 'q' to stop.")
        frame_idx = 0
        
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            
            # -----------------------------------------------------------------
            # 4. PROCESSING LOOP
            # -----------------------------------------------------------------
            # Align first
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
                
            # Filter Pipeline
            # 1. Depth -> Disparity
            frame = depth_to_disparity.process(aligned_depth_frame)
            
            # 2. Spatial Filter (Smooths geometry)
            frame = spatial.process(frame)
            
            # 3. Disparity -> Depth
            frame = disparity_to_depth.process(frame)
            
            # 4. Temporal Filter (Removes Jitter/Flicker - Must run on Depth)
            frame = temporal.process(frame)
            
            # 5. Hole Filling (Fills gaps)
            frame = hole_filling.process(frame)
            
            # Convert to numpy
            depth_image = np.asanyarray(frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Save frames
            cv2.imwrite(os.path.join(rgb_dir, f"{frame_idx:06d}.jpg"), color_image)
            cv2.imwrite(os.path.join(depth_dir, f"{frame_idx:06d}.png"), depth_image)
            
            # Visualization
            # Scale depth for visualization (0.03 is strictly for viewing, doesn't affect saved data)
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
