# benchmark_recorder.py

import os
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Set

# Define the type for the dictionary that will hold the current frame's GT data
# {gt_id: (cx, cy, w, h)}
GTBoxesType = Dict[int, Tuple[float, float, float, float]]

class BenchmarkRecorder:
    
    # Define a default persistence threshold
    DEFAULT_PERSISTENCE_THRESHOLD = 5 
    
    def __init__(self, tracker_name, persistence_threshold: int = DEFAULT_PERSISTENCE_THRESHOLD):
        """
        Initializes CSV loggers and the state trackers for GT persistence.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/{tracker_name}_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # --- NEW PERSISTENCE STATE ---
        self.gt_persistence_tracker: Dict[int, int] = {} # {gt_id: consecutive_frame_count}
        self.persistence_threshold = persistence_threshold
        print(f"Filter: GT boxes must be present for >= {self.persistence_threshold} frames to be logged.")
        
        # --- SETUP TRACKER LOG ---
        self.track_file_path = os.path.join(self.log_dir, "tracker_output.csv")
        self.track_file = open(self.track_file_path, 'w', newline='')
        self.track_writer = csv.writer(self.track_file)
        self.track_writer.writerow(["FrameId", "Id", "X", "Y", "W", "H", "conf", "latency_ms"])

        # --- SETUP GROUND TRUTH LOG ---
        self.gt_file_path = os.path.join(self.log_dir, "ground_truth.csv")
        self.gt_file = open(self.gt_file_path, 'w', newline='')
        self.gt_writer = csv.writer(self.gt_file)
        self.gt_writer.writerow(["FrameId", "Id", "X", "Y", "W", "H"])

        print(f"📂 Logging data to: {self.log_dir}")


    def log(self, frame_id, track_id, cx, cy, w, h, conf, latency):
        """Logs a single tracker prediction (Center X, Center Y)."""
        self.track_writer.writerow([
            frame_id, 
            track_id, 
            f"{cx:.2f}",
            f"{cy:.2f}",
            f"{w:.2f}", 
            f"{h:.2f}", 
            f"{conf:.4f}", 
            f"{latency:.2f}"
        ])

    
    # The original log_gt is replaced by this stateful method
    def update_persistence_and_log_gt(self, frame_id: int, current_gt_boxes: GTBoxesType) -> GTBoxesType:
        """
        Handles persistence tracking, filters out transient boxes, and logs only the stable GT boxes.
        
        Parameters:
            frame_id: The current frame number.
            current_gt_boxes: The dict of {gt_id: (cx, cy, w, h)} for the current frame.
            
        Returns:
            A dictionary of {gt_id: (cx, cy, w, h)} containing ONLY the boxes 
            that have met the persistence threshold (for drawing/visualization).
        """
        current_ids: Set[int] = set(current_gt_boxes.keys())
        logged_boxes: GTBoxesType = {}
        
        # --- 1. UPDATE COUNTS (Increment for present IDs) ---
        for gt_id in current_ids:
            # If present, increment count
            self.gt_persistence_tracker[gt_id] = self.gt_persistence_tracker.get(gt_id, 0) + 1
        
        # --- 2. DECAY/REMOVE STALE IDs ---
        ids_to_remove: List[int] = []
        # Iterate over a copy of keys to safely modify the dictionary
        for stored_id in list(self.gt_persistence_tracker.keys()):
            if stored_id not in current_ids:
                # Decay the count if the ID is missing in the current frame
                self.gt_persistence_tracker[stored_id] -= 1
                
                # Remove the ID if its count drops to zero
                if self.gt_persistence_tracker[stored_id] <= 0:
                    ids_to_remove.append(stored_id)

        for stale_id in ids_to_remove:
            del self.gt_persistence_tracker[stale_id]

        # --- 3. LOG PERSISTENT BOXES & RETURN FOR DRAWING ---
        for gt_id, box in current_gt_boxes.items():
            cx, cy, w, h = box
            
            # Check if the ID has met the persistence threshold
            if self.gt_persistence_tracker.get(gt_id, 0) >= self.persistence_threshold:
                
                # Write to the ground_truth.csv file
                self.gt_writer.writerow([
                    frame_id, 
                    gt_id, 
                    f"{cx:.2f}",
                    f"{cy:.2f}",
                    f"{w:.2f}", 
                    f"{h:.2f}"
                ])
                
                # Store the box for the main loop to draw
                logged_boxes[gt_id] = box 
                
        return logged_boxes


    def close(self):
        """Closes the file streams safely."""
        self.track_file.close()
        self.gt_file.close()
        print(f"💾 Logs saved. Path: {self.log_dir}")