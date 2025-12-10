import sys
import os
import open3d as o3d
# Add Grounded-SAM-2 to path to handle internal imports like 'grounding_dino.groundingdino'
sys.path.append(os.path.join(os.getcwd(), "Grounded-SAM-2"))

import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Internal imports from Grounding DINO
try:
    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
except ImportError:
    # Fallback if installed as standard package but code uses relative
    from groundingdino.util.inference import load_model, load_image, predict

# Constants
SAM2_CHECKPOINT = "./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "./Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = "./Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Load Data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image (jpg/png)")
    parser.add_argument("--output", type=str, default="output_segmentation.jpg", help="Output image path")
    args = parser.parse_args()
    
    if not os.path.exists(args.rgb):
        print(f"Error: {args.rgb} not found.")
        return
    
    # Load RGB image
    color_image = cv2.imread(args.rgb)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    print(f"Loaded RGB: {args.rgb}")

    # Save RGB to temp file for G-DINO load_image
    temp_img_path = "temp_input.jpg"
    cv2.imwrite(temp_img_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

    # 2. Load Models
    print("Loading SAM 2...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading Grounding DINO...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    # 3. Predict
    text_prompt = input("Enter text prompt (default: 'mug .'): ") or "mug ."
    
    print(f"Running detection for: '{text_prompt}'")
    image_source, image = load_image(temp_img_path)
    
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )
    
    print(f"Detected {len(boxes)} objects.")

    if len(boxes) == 0:
        print("No objects detected.")
        return

    # Process boxes for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # Filter out boxes that contain other boxes (nested detection removal)
    keep_indices = []
    for i in range(len(input_boxes)):
        is_container = False
        box_a = input_boxes[i]
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        
        for j in range(len(input_boxes)):
            if i == j:
                continue
            
            box_b = input_boxes[j]
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            
            # Intersection
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])
            
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            intersection = inter_w * inter_h
            
            # If Box A contains > 90% of Box B, and Box A is larger than Box B
            if intersection > 0.9 * area_b and area_a > area_b:
                is_container = True
                break
        
        if not is_container:
            keep_indices.append(i)
    
    input_boxes = input_boxes[keep_indices]
    confidences = confidences[keep_indices]
    # Handle labels if it's a list or tensor
    if isinstance(labels, list):
        labels = [labels[i] for i in keep_indices]
    else:
        labels = labels[keep_indices]
    
    print(f"Filtered to {len(input_boxes)} objects (removed containers).")
    
    if len(input_boxes) == 0:
        print("No objects left after filtering.")
        return

    # SAM 2 Inference
    print("Running SAM 2 segmentation...")
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
         masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 4. Visualize and Save
    print("Creating visualization...")
    
    # Create mask overlay
    # masks shape: (N, H, W)
    if masks.shape[0] > 0:
        combined_mask = np.any(masks, axis=0)  # (H, W) boolean
    else:
        combined_mask = np.zeros(color_image.shape[:2], dtype=bool)

    # Create visualization with semi-transparent overlay
    viz_image = color_image.copy()
    
    # Create a colored overlay (green with 50% transparency)
    overlay = viz_image.copy()
    overlay[combined_mask] = [0, 255, 0]  # Green
    viz_image = cv2.addWeighted(viz_image, 0.7, overlay, 0.3, 0)
    
    # Draw bounding boxes
    for box in input_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(viz_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red boxes
    
    # Save output
    output_bgr = cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output_bgr)
    print(f"Saved segmentation result to: {args.output}")

if __name__ == "__main__":
    main()

