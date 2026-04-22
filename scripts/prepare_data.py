import sqlite3
import os
import shutil
import random
from pathlib import Path

def normalize_bbox(bbox, img_width=800, img_height=600):
    """Convert (x_tl, y_tl, x_br, y_br) to YOLO (x_center, y_center, w, h) normalized."""
    x_tl, y_tl, x_br, y_br = bbox
    
    # Calculate width and height
    w = x_br - x_tl
    h = y_br - y_tl
    
    # Calculate center
    x_center = x_tl + w / 2
    y_center = y_tl + h / 2
    
    # Normalize
    return (x_center / img_width, y_center / img_height, w / img_width, h / img_height)

def main():
    # Paths
    db_path = "dataset/sherbrooke_annotations/sherbrooke_gt.sqlite"
    frames_dir = "dataset/sherbrooke_frames"
    output_base = "datasets/traffic"
    
    # Class mapping: DB -> YOLO
    # DB: 1: car, 2: pedestrians, 3: motorcycle, 4: bicycle
    # YOLO: 0: person, 1: bicycle, 2: car, 3: motorcycle
    class_map = {
        1: 2, # car
        2: 0, # pedestrians -> person
        3: 3, # motorcycle
        4: 1  # bicycle
    }

    # Range of annotated frames
    start_frame = 2754
    end_frame = 3754
    
    # Split ratio
    train_ratio = 0.8
    
    # Create directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_base, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_base, 'labels', split), exist_ok=True)

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all bounding boxes for the annotated range
    print(f"Fetching annotations from frame {start_frame} to {end_frame}...")
    
    # We need to join bounding_boxes with objects to get the road_user_type
    query = """
    SELECT b.frame_number, o.road_user_type, b.x_top_left, b.y_top_left, b.x_bottom_right, b.y_bottom_right
    FROM bounding_boxes b
    JOIN objects o ON b.object_id = o.object_id
    WHERE b.frame_number BETWEEN ? AND ?
    """
    cursor.execute(query, (start_frame, end_frame))
    all_data = cursor.fetchall()
    
    # Group by frame
    frame_annotations = {}
    for row in all_data:
        frame, obj_type, x1, y1, x2, y2 = row
        if obj_type not in class_map:
            continue
            
        yolo_class = class_map[obj_type]
        yolo_bbox = normalize_bbox((x1, y1, x2, y2))
        
        if frame not in frame_annotations:
            frame_annotations[frame] = []
        frame_annotations[frame].append((yolo_class, *yolo_bbox))
    
    conn.close()
    
    # List of frames to process
    frames = sorted(frame_annotations.keys())
    random.seed(42)
    random.shuffle(frames)
    
    split_idx = int(len(frames) * train_ratio)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]
    
    print(f"Total annotated frames: {len(frames)}")
    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")
    
    # Process splits
    for split_name, frame_list in [('train', train_frames), ('val', val_frames)]:
        print(f"Processing {split_name} set...")
        for frame_num in frame_list:
            # Source image path (e.g. 00002754.jpg)
            img_name = f"{frame_num:08d}.jpg"
            src_img = os.path.join(frames_dir, img_name)
            
            if not os.path.exists(src_img):
                print(f"Warning: Frame {img_name} not found in {frames_dir}")
                continue
                
            # Target paths
            dst_img = os.path.join(output_base, 'images', split_name, img_name)
            dst_lbl = os.path.join(output_base, 'labels', split_name, f"{frame_num:08d}.txt")
            
            # Copy image
            shutil.copy(src_img, dst_img)
            
            # Write labels
            with open(dst_lbl, "w") as f:
                for ann in frame_annotations[frame_num]:
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
                    
    print("\nDataset preparation completed successfully!")
    print(f"Dataset location: {os.path.abspath(output_base)}")

if __name__ == "__main__":
    main()
