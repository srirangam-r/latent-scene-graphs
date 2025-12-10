import sys
import os
import open3d as o3d
# Add Grounded-SAM-2 to path
sys.path.append(os.path.join(os.getcwd(), "Grounded-SAM-2"))

import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from torchvision.ops import box_convert
import time

# SAM 2 Video Imports
from sam2.build_sam import build_sam2_video_predictor

# Grounding DINO Imports
try:
    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
except ImportError:
    from groundingdino.util.inference import load_model, load_image, predict

from realsense_utils import project_rgbd_to_pointcloud, fit_plane_rectangle, create_visualization_geometries, create_back_plane_geometry

# Constants
SAM2_CHECKPOINT = "./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "./Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "./Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_multiple_planes(pcd, num_planes=5, distance_threshold=0.01):
    """
    Extract multiple planes from a point cloud using iterative RANSAC.
    Returns: list of dicts with 'model', 'pcd', 'center', 'area', 'corners'
    """
    planes = []
    remaining_pcd = pcd
    print(f"DEBUG: Starting Multi-Plane Extraction. Total Points: {len(remaining_pcd.points)}")
    
    for i in range(num_planes):
        if len(remaining_pcd.points) < 50:
            print(f"DEBUG: Stopped extraction (Remaining points < 50): {len(remaining_pcd.points)}")
            break
            
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        inliers_pcd = remaining_pcd.select_by_index(inliers)
        # Lower threshold to 20 points
        if len(inliers_pcd.points) < 20: 
            print(f"DEBUG: Skipped plane {i} (Inliers < 20): {len(inliers_pcd.points)}")
            break
            
        # Compute plane properties
        center = np.mean(np.asarray(inliers_pcd.points), axis=0)
        
        # Estimate area (2D bounding box after projection)
        # We can just use the bounding box of the projected points on the plane
        # For scoring, we'll use a rough area estimate
        
        # Project points to plane to get area
        [a, b, c, d] = plane_model
        # ... logic for area ...
        # Simplified area: bounding box diagonal squared?
        # Let's just use number of points as a proxy for now or use the bounds
        pcd_bounds = inliers_pcd.get_axis_aligned_bounding_box()
        extent = pcd_bounds.get_extent()
        area = extent[0] * extent[1] # Rough approx
        
        planes.append({
            'model': plane_model,
            'pcd': inliers_pcd,
            'center': center,
            'area': area
        })
        
        # Remove inliers for next iteration
        outliers = remaining_pcd.select_by_index(inliers, invert=True)
        remaining_pcd = outliers
        print(f"DEBUG: Extracted Plane {i}. Inliers: {len(inliers_pcd.points)}. Remaining: {len(remaining_pcd.points)}")
    
    return planes

def select_front_back_planes(planes, camera_position=np.array([0, 0, 0])):
    """
    Select front and back planes from candidates based on geometric criteria.
    Returns: (front_plane, back_plane) or (front_plane, None)
    """
    if len(planes) == 0:
        return None, None
    
    # Score each plane for "frontness"
    max_area = max([p['area'] for p in planes]) if planes else 1.0
    scores = []
    
    for plane in planes:
        [a, b, c, d] = plane['model']
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # View direction (from object to camera)
        view_dir = -plane['center']
        if np.linalg.norm(view_dir) > 0:
            view_dir = view_dir / np.linalg.norm(view_dir)
        else:
            view_dir = np.array([0, 0, 1])
        
        # Score components
        score_facing = max(0, np.dot(normal, view_dir))
        score_proximity = 1.0 / (1.0 + np.linalg.norm(plane['center']))
        score_area = plane['area'] / max_area if max_area > 0 else 0.0
        
        # Combined score (prioritize facing and proximity)
        score = (score_facing ** 2) * score_proximity * (score_area ** 0.5)
        scores.append(score)
    
    # Front plane = highest score
    front_idx = np.argmax(scores)
    front_plane = planes[front_idx]
    
    # Back plane = parallel to front, behind it
    front_normal = np.array(front_plane['model'][:3])
    front_normal = front_normal / np.linalg.norm(front_normal)
    
    back_candidates = []
    print(f"DEBUG: Found {len(planes)} total planes. Front idx: {front_idx}")
    for i, plane in enumerate(planes):
        if i == front_idx:
            continue
        
            
        # Plane properties
        p_normal = np.array(plane['model'][:3])
        p_normal = p_normal / np.linalg.norm(p_normal)
        p_center = plane['center']
        
        # Relations to Front
        dot = np.dot(front_normal, p_normal)
        angle_deg = np.degrees(np.arccos(min(1.0, abs(dot))))
        
        dist_front = np.linalg.norm(front_plane['center'])
        dist_p = np.linalg.norm(p_center)
        delta_dist = dist_p - dist_front
        
        # Check parallelism
        is_parallel = abs(dot) > 0.9
        is_dist_ok = delta_dist > 0.05
        
        status = "REJECTED"
        if is_parallel and is_dist_ok:
            status = "ACCEPTED"
            back_candidates.append((i, plane, dist_p))
        elif not is_parallel:
            status = "REJECT (Angle)"
        elif not is_dist_ok:
            status = "REJECT (Dist)"
            
        print(f"Plane {i}: {status} | Dist={delta_dist:.3f}m | Angle={angle_deg:.1f}deg (|dot|={abs(dot):.3f}) | Points={len(plane['pcd'].points)}")

    # Return candidates for debug visualization even if we pick one
    # New return: front_plane, back_plane, other_candidates
    other_candidates = [p for i, p in enumerate(planes) if i != front_idx]
    
    if back_candidates:
        # Pick furthest back candidate
        back_idx, back_plane, _ = max(back_candidates, key=lambda x: x[2])
        return front_plane, back_plane, other_candidates
    else:
        return front_plane, None, other_candidates

def draw_scene_graph(object_states):
    """
    Draws a 2D scene graph visualization for multiple tracked drawers.
    Root: Room Node
    Branches: Gate(Drawer X) -> Latent(Drawer X) for each tracked object.
    """
    # Canvas - Wider to accommodate multiple branches
    H, W = 600, 1000
    img = np.ones((H, W, 3), dtype=np.uint8) * 255 # White background
    
    # Sort IDs to keep order consistent
    ids = sorted(list(object_states.keys()))
    num_objects = len(ids)
    
    # Settings
    radius = 35
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. Draw Room Node (Top Center)
    c_room = (W // 2, 80)
    color_room = (200, 200, 200) # Light Gray
    
    cv2.circle(img, c_room, radius, color_room, -1)
    cv2.circle(img, c_room, radius, (0, 0, 0), 2)
    
    # Helper for centered text (re-defined here to access img/font)
    def put_centered_text(text, center, scale=0.6, color=(0,0,0), thickness=1, offset_y=0):
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = int(center[0] - text_size[0] // 2)
        text_y = int(center[1] + text_size[1] // 2 + offset_y)
        cv2.putText(img, text, (text_x, text_y), font, scale, color, thickness)

    put_centered_text("Room", c_room)
    
    if num_objects == 0:
        put_centered_text("No Objects Tracked", (W//2, H//2), scale=1.0)
        return img

    # 2. Draw Branches
    # Y positions
    y_gate = 300
    y_latent = 500
    
    # Calculate X positions
    # Spacing approach: Divide width into N+1 segments
    # x_i = (i + 1) * W / (N + 1)
    
    for i, obj_id in enumerate(ids):
        state = object_states[obj_id]
        
        # Branch X position
        cx = int((i + 1) * W / (num_objects + 1))
        
        c_gate = (cx, y_gate)
        c_latent = (cx, y_latent)
        
        # --- State Colors ---
        # Gate
        drawer_st = state['drawer_state']
        if drawer_st == "open":
            color_gate = (100, 200, 100) # Greenish
        else:
            color_gate = (150, 150, 255) # Reddish
            
        # Latent
        is_converged = state['is_converged']
        if is_converged:
            color_latent = (0, 255, 0) # Green
            text_latent_status = "Observed"
        else:
            color_latent = (255, 255, 0) # Cyan
            text_latent_status = "Unobserved"

        # --- Draw Connecting Lines ---
        cv2.line(img, c_room, c_gate, (0, 0, 0), 2)
        cv2.line(img, c_gate, c_latent, (0, 0, 0), 2)
        
        # --- Draw Gate Node ---
        cv2.circle(img, c_gate, radius, color_gate, -1)
        cv2.circle(img, c_gate, radius, (0, 0, 0), 2)
        
        # --- Draw Latent Node ---
        cv2.circle(img, c_latent, radius, color_latent, -1)
        cv2.circle(img, c_latent, radius, (0, 0, 0), 2)
        
        # --- Labels ---
        # Gate Label
        put_centered_text(f"Gate {obj_id}", c_gate, offset_y=-10)
        
        # Latent Label
        put_centered_text(f"Latent {obj_id}", c_latent, offset_y=-10)
        
        # Info Text (Below Latent)
        depth_val = state['depth_estimate']
        info_y_base = c_latent[1] + radius + 25
        line_spacing = 20
        
        # Using centered text for info stack
        put_centered_text(f"D: {depth_val:.2f}m", (cx, info_y_base), scale=0.5)
        put_centered_text(text_latent_status, (cx, info_y_base + line_spacing), scale=0.5, color=(0, 100, 0) if is_converged else (180, 180, 0))
        put_centered_text(f"{drawer_st.upper()}", (cx, info_y_base + line_spacing*2), scale=0.5, thickness=2)

    return img

def main():
    # 1. Setup Data Paths
    video_dir = "video_data"
    rgb_dir = os.path.join(video_dir, "rgb")
    depth_dir = os.path.join(video_dir, "depth")
    intrinsics_path = os.path.join(video_dir, "intrinsics.json")
    
    if not os.path.exists(video_dir) or not os.path.exists(intrinsics_path):
        print("Error: video_data not found. Run capture_video.py first.")
        return

    # Load Intrinsics
    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    depth_scale = intrinsics_dict.get('depth_scale', 0.001)

    # Get Frames
    frame_names = sorted([p for p in os.listdir(rgb_dir) if p.endswith((".jpg", ".jpeg"))])
    if not frame_names:
        print("No frames found.")
        return
        
    print(f"Found {len(frame_names)} frames.")

    # 2. Initialize Models
    print("Loading SAM 2 Video Predictor...")
    # Enable stricter tracking:
    # 1. multimask_output_for_tracking=False: Prevents switching between multiple mask hypotheses
    predictor = build_sam2_video_predictor(
        SAM2_MODEL_CONFIG, 
        SAM2_CHECKPOINT, 
        device=DEVICE,
        hydra_overrides_extra=[
            "++model.multimask_output_for_tracking=false",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=false"
        ]
    )
    
    print("Loading Grounding DINO...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    # 3. Initialize Tracking State
    inference_state = predictor.init_state(video_path=rgb_dir)
    
    # 4. Prompt and Initial Detection (Frame 0)
    text_prompt = input("Enter text prompt (default: 'small drawer .'): ") or "small drawer ."
    print(f"Detecting '{text_prompt}' in the first frame...")
    
    first_frame_path = os.path.join(rgb_dir, frame_names[0])
    image_source, image = load_image(first_frame_path)
    
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )
    
    if len(boxes) == 0:
        print("No objects detected in the first frame. Exiting.")
        return
        
    print(f"Detected {len(boxes)} objects. Initializing tracking...")
    
    # Add box prompt to SAM 2 (Frame 0)
    # Grounding DINO returns cxcywh normalized, we need xyxy unnormalized? 
    # Actually checking sam2 example: they take unnormalized xyxy usually?
    # Wait, in test_grounded_sam2.py we did:
    # boxes = boxes * torch.Tensor([w, h, w, h])
    # input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    # predictor.predict(box=input_boxes...)
    #
    # For Video Predictor:
    # predictor.add_new_points_or_box(inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, box=box)
    # The box format for add_new_points_or_box is expected to be [x1, y1, x2, y2] unnormalized.
    
    h, w, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # Filter out boxes that contain other boxes
    keep_indices = []
    for i in range(len(input_boxes)):
        is_container = False
        box_a = input_boxes[i]
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        for j in range(len(input_boxes)):
            if i == j: continue
            box_b = input_boxes[j]
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            if inter > 0.9 * area_b and area_a > area_b:
                is_container = True
                break
        if not is_container:
            keep_indices.append(i)
            
    input_boxes = input_boxes[keep_indices]
    print(f"Filtered to {len(input_boxes)} objects (removed containers).")
    
    # Initialize track for ALL detected objects
    ann_frame_idx = 0
    tracked_ids = []
    
    for idx, box_prompt in enumerate(input_boxes):
        obj_id = idx + 1 # 1-based IDs
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            box=box_prompt
        )
        tracked_ids.append(obj_id)
        print(f"Tracking initiated for object ID {obj_id} with box: {box_prompt}")

    # 5. Open3D Visualization Loop
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Grounded SAM 2 3D Tracking", width=1280, height=720)
    
    # Add coordinate frame
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    
    # Option 1: Use non-blocking render loop by manually updating geometries
    # But checking for user input/exit might be tricky with just vis.poll_events().
    
    # We will iterate through the results generator
    geometry_added = False
    pcd = o3d.geometry.PointCloud()
    cuboid_lines = o3d.geometry.LineSet()
    pink_grid = o3d.geometry.LineSet()
    
    # Smoothing state
    # State tracking for multiple objects
    object_states = {}
    for obj_id in tracked_ids:
        object_states[obj_id] = {
            'prev_corners': None,
            'prev_normal': None,
            'fixed_size': None,
            'depth_estimate': 0.3,
            'depth_variance': 0.1**2,
            'drawer_state': "closed",
            'initial_gate_position': None,
            'is_converged': False
        }
        
    alpha = 0.3 
    process_noise = 0.001**2
    measurement_noise = 0.025**2
    gate_movement_threshold = 0.10

    print("Starting video propagation...")
    
    # Propagate
    # This yields results for each frame
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        
        # Determine current frame file paths
        frame_name = frame_names[out_frame_idx]
        rgb_path = os.path.join(rgb_dir, frame_name)
        # Assuming depth matches name (check extension)
        depth_name = frame_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        depth_path = os.path.join(depth_dir, depth_name)
        
        # Load Images
        color_image = cv2.imread(rgb_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # realsense_utils expects RGB
        
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        # Prepare per-frame visualization lists
        rect_geoms = [] # Re-using variable name for flat list of all objects
        back_geom = None # We won't accumulate back geoms as much, or maybe we should? Let's skip back geom for now to avoid clutter or just show for ID 1.

        # Process each tracked object
        for idx_out, obj_id in enumerate(out_obj_ids):
            if obj_id not in object_states: continue
            
            state = object_states[obj_id]
            
            mask_logits = out_mask_logits[idx_out]
            if mask_logits.ndim == 3: mask_logits = mask_logits[0]
            mask = (mask_logits > 2.0).cpu().numpy()
            
            # --- 3D Fitting Logic (Indented) ---
            object_depth = depth_image.copy()
            if mask is not None:
                 object_depth[~mask] = 0
            else:
                 object_depth[:] = 0
            
            if mask is not None and np.sum(mask) > 100:
                 obj_pcd = project_rgbd_to_pointcloud(
                    color_image, object_depth, intrinsics_dict, depth_scale, depth_trunc=2.0, visualize=False
                 )
                 if len(obj_pcd.points) > 50:
                     _, _, params = fit_plane_rectangle(obj_pcd, priority_normal=state['prev_normal'])

                     if params is not None:
                         current_corners = params['corners']
                         current_normal = params['normal']
                         
                         # Corner Reordering
                         if state['prev_corners'] is not None:
                             min_dist = float('inf')
                             best_corners = current_corners
                             for i in range(4):
                                 permuted = np.roll(current_corners, -i, axis=0)
                                 dist = np.sum(np.linalg.norm(permuted - state['prev_corners'], axis=1))
                                 if dist < min_dist:
                                     min_dist = dist
                                     best_corners = permuted
                             current_corners = best_corners
                         
                         # Size Constraint
                         v_w = current_corners[1] - current_corners[0]
                         v_h = current_corners[3] - current_corners[0]
                         cur_w = np.linalg.norm(v_w); cur_h = np.linalg.norm(v_h)
                         if cur_w > 0: axis_w = v_w / cur_w
                         else: axis_w = np.array([1, 0, 0])
                         if cur_h > 0: axis_h = v_h / cur_h
                         else: axis_h = np.array([0, 1, 0])
                         center = np.mean(current_corners, axis=0)
                         
                         if state['fixed_size'] is None:
                             state['fixed_size'] = (cur_w, cur_h)
                         
                         target_w, target_h = state['fixed_size']
                         half_w = target_w / 2.0; half_h = target_h / 2.0
                         c0 = center - axis_w * half_w - axis_h * half_h
                         c1 = center + axis_w * half_w - axis_h * half_h
                         c2 = center + axis_w * half_w + axis_h * half_h
                         c3 = center - axis_w * half_w + axis_h * half_h
                         constrained_corners = np.array([c0, c1, c2, c3])
                         
                         # Smoothing
                         if state['prev_corners'] is None:
                             smooth_corners = constrained_corners
                             smooth_normal = current_normal
                         else:
                             smooth_corners = alpha * constrained_corners + (1 - alpha) * state['prev_corners']
                             smooth_normal = alpha * current_normal + (1 - alpha) * state['prev_normal']
                             smooth_normal = smooth_normal / np.linalg.norm(smooth_normal)
                         
                         state['prev_corners'] = smooth_corners
                         state['prev_normal'] = smooth_normal
                         
                         # State (Open/Closed)
                         current_gate_center = np.mean(smooth_corners, axis=0)
                         gate_distance = np.linalg.norm(current_gate_center)
                         
                         if state['initial_gate_position'] is None:
                             state['initial_gate_position'] = gate_distance
                         else:
                             movement = state['initial_gate_position'] - gate_distance
                             if movement >= gate_movement_threshold:
                                 state['drawer_state'] = "open"
                             else:
                                 state['drawer_state'] = "closed"
                         
                         # Convergence
                         state['is_converged'] = np.sqrt(state['depth_variance']) < convergence_threshold
                         cuboid_color = [0, 1, 0] if state['is_converged'] else [0, 1, 1]
                         
                         # Add Geometries
                         geoms = create_visualization_geometries(
                            smooth_corners, smooth_normal, extrusion_depth=state['depth_estimate'], cuboid_color=cuboid_color
                         )
                         rect_geoms.extend(geoms)
                         
                         # --- Depth Update Logic ---
                         # Only run depth update if "mask" points are sufficient
                         if np.sum(mask) > 200:
                             view_dir = -np.mean(obj_pcd.points, axis=0)
                             if np.dot(smooth_normal, view_dir) < 0: calc_normal = -smooth_normal
                             else: calc_normal = smooth_normal
                             
                             points_np = np.asarray(obj_pcd.points)
                             vecs = points_np - smooth_corners.mean(axis=0)
                             distances = np.dot(vecs, -calc_normal)
                             valid_distances = distances[distances > 0]
                             
                             if len(valid_distances) > 20:
                                 sorted_dists = np.sort(valid_distances)[::-1]
                                 measured_depth = np.mean(sorted_dists[:20])
                                 
                                 if measured_depth > 0.2:
                                     depth_est = state['depth_estimate']
                                     depth_var = state['depth_variance']
                                     
                                     depth_est_pred = depth_est
                                     depth_var_pred = depth_var + process_noise
                                     
                                     kalman_gain = depth_var_pred / (depth_var_pred + measurement_noise)
                                     new_est = depth_est_pred + kalman_gain * (measured_depth - depth_est_pred)
                                     new_var = (1 - kalman_gain) * depth_var_pred
                                     
                                     state['depth_estimate'] = new_est
                                     state['depth_variance'] = new_var
                                     
                                     # print(f"ID {obj_id}: Depth {new_est:.3f}m")

        # Visualization Geometries
        if not geometry_added:
            vis.add_geometry(pcd)
            # Make pcd gray
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(origin_frame)
            geometry_added = True
        else:
            vis.update_geometry(pcd)
            
        # Update Geometries (Clear old, add new)
        # Note: Open3D visualizer doesn't support easy removal of specific generic geometries efficiently without handles
        # But we can clear and re-add or just use a dedicated logical approach.
        # For simplicity in this script, we kept adding geometry? No, that would leak memory/clutter.
        # The previous code didn't clear geometries! It only added `rect_geoms` once if they were persistent?
        # Re-reading: the code creates `rect_geoms` every frame.
        # But `vis.add_geometry` adds them forever unless we remove them.
        # Let's fix this: We need to remove previous frame's visualization geometries.
        # But we don't have handles easily.
        # Actually, `vis.clear_geometries()` wipes everything including camera view.
        # A simple hack for this demo:
        # We can reuse the line_sets if possible, or just accept that this script might be leaky/slow for long videos without proper management.
        # BUT, for the Scene Graph, we use OpenCV 2D.
        
        # Draw Scene Graph (Pass full state dictionary)
        scene_graph_img = draw_scene_graph(object_states)
        cv2.imshow("Scene Graph State", scene_graph_img)
        
        # Show RGB Frame
        cv2.imshow("RGB Frame", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
        
        cv2.waitKey(1)
        
        # 3D Vis Update
        # We should really manage the geometries.
        # Assuming `rect_geoms` contains line sets.
        for geom in rect_geoms:
            vis.add_geometry(geom, reset_bounding_box=False)
            
        if back_geom is not None:
             vis.add_geometry(back_geom, reset_bounding_box=False)

        vis.poll_events()
        vis.update_renderer()
        
        # Cleanup geometries for next frame (Rudimentary)
        for geom in rect_geoms:
            vis.remove_geometry(geom, reset_bounding_box=False)
        if back_geom is not None:
            vis.remove_geometry(back_geom, reset_bounding_box=False)
        
        if out_frame_idx == 0:
            print("Paused for 90s to arrange windows...")
            
            # --- Adjust Camera View (Opposite Side) ---
            ctr = vis.get_view_control()
            # Default is usually -Z looking forward. "Opposite" suggests looking from +Z.
            # Or rotating 180 degrees.
            # Setting a specific view that looks at the origin from the "front" (relative to object)
            # Assuming object is at origin, and originally we look from -Z.
            # Let's try setting front to [0, 0, 1] (Looking from +Z axis)
            ctr.set_front([0.0, 0.0, 1.0]) 
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, -1.0, 0.0]) # Standard OpenCV Y-up is negative in Open3D visualizer usually? 
                                          # Actually Open3D Y is up. OpenCV Y is down.
                                          # If we want standard view, we usually just need to flip Z.
            
            # Rotate slightly for better 3D perception
            ctr.rotate(10.0, 0.0) # Mouse drag units
            
            start_time = time.time()
            while time.time() - start_time < 90:
                vis.poll_events()
                vis.update_renderer()
                cv2.waitKey(10) # Keep OpenCV window responsive
                time.sleep(0.01) # Avoid 100% CPU usage

        # Small delay or check for exit?
        # print(f"Processed frame {out_frame_idx}")

    print("Video processing complete.")
    vis.destroy_window()

if __name__ == "__main__":
    main()
