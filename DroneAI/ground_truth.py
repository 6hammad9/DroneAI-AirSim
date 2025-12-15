import numpy as np
import cv2

def decode_segmentation(response):
    """
    Converts AirSim ImageResponse to a 2D array of Object IDs.
    """
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    
    img_rgb = None
    if img1d.size == response.width * response.height * 4:
        img_rgb = img1d.reshape(response.height, response.width, 4)[:, :, :3]
    elif img1d.size == response.width * response.height * 3:
        img_rgb = img1d.reshape(response.height, response.width, 3)
    else:
        return np.zeros((response.height, response.width), dtype=np.int32)

    img_rgb = img_rgb.astype(np.uint32)
    seg_id_map = img_rgb[:, :, 0] + (img_rgb[:, :, 1] << 8) + (img_rgb[:, :, 2] << 16)
    return seg_id_map


def extract_gt_boxes(seg_id_map, frame_width=640, frame_height=480,
                     min_w=3, min_h=3, min_area=20, 
                     max_w_ratio=0.95, max_h_ratio=0.95,
                     max_aspect_ratio=15.0, min_aspect_ratio=0.1,
                     scale_with_distance=False, camera_pos=None, object_depth_map=None):
    
    gt_boxes = {}
    unique_ids = np.unique(seg_id_map)

    # Kernel for morphological operations (3x3 is standard for cleaning)
    kernel = np.ones((3, 3), np.uint8) 

    for obj_id in unique_ids:
        if obj_id == 0:
            continue

        mask = (seg_id_map == obj_id).astype(np.uint8) * 255

        # =========================================================
        # ✨ MORPHOLOGICAL CLEANING CODE ADDED HERE
        # =========================================================
        # 1. MORPH_CLOSE: Fills small gaps/holes inside the object mask. 
        #    This prevents a single object from being split into multiple components.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 2. MORPH_OPEN: Removes tiny isolated specks (noise) that are too small 
        #    to be considered a connected component, helping the min_area filter.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # =========================================================

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # --- FILTER 1: TINY NOISE (If noise survives cleaning, this catches it) ---
            if w < min_w or h < min_h or area < min_area:
                continue

            # --- FILTER 2: GIANT BACKGROUND ---
            if w > (frame_width * max_w_ratio) or h > (frame_height * max_h_ratio):
                continue

            # --- FILTER 3: ASPECT RATIO ---
            aspect_ratio = w / h
            if aspect_ratio > max_aspect_ratio or aspect_ratio < min_aspect_ratio:
                continue

            # Unique ID generation for separate components/objects
            unique_sub_id = int(f"{obj_id}{i}") 

            cx = x + w / 2
            cy = y + h / 2

            # --- SCALE BOX BASED ON DISTANCE ---
            if scale_with_distance and camera_pos is not None:
                if object_depth_map and unique_sub_id in object_depth_map:
                    distance = object_depth_map[unique_sub_id]
                else:
                    distance = 1.0
                scale = max(0.1, min(1.0, 50.0 / distance))
                w = int(w * scale)
                h = int(h * scale)

            gt_boxes[unique_sub_id] = (cx, cy, w, h)

    return gt_boxes