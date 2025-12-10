import os
import glob

def reduce_framerate(keep_every=4):
    video_dir = "video_data"
    rgb_dir = os.path.join(video_dir, "rgb")
    depth_dir = os.path.join(video_dir, "depth")
    
    # Get all files sorted
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    
    print(f"Original: {len(rgb_files)} RGB, {len(depth_files)} Depth")
    
    # Verify sync
    if len(rgb_files) != len(depth_files):
        print("Warning: RGB and Depth counts differ!")
        
    files_to_remove = 0
    
    for i, (rgb_im, depth_im) in enumerate(zip(rgb_files, depth_files)):
        if i % keep_every != 0:
            os.remove(rgb_im)
            os.remove(depth_im)
            files_to_remove += 1
            
    print(f"Removed {files_to_remove} frame pairs.")
    
    # Check remaining
    rgb_files_new = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    print(f"Remaining: {len(rgb_files_new)} pairs.")

if __name__ == "__main__":
    reduce_framerate()
