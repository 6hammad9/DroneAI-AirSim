# main.py
import cv2
import numpy as np
import time

from drone_control import DroneController
from vision import VisionSystem
from tracking_system import DroneTracker
from benchmark_recorder import BenchmarkRecorder
import config

from ground_truth import decode_segmentation, extract_gt_boxes


def get_color(id):
    """Generates a consistent unique color for each ID"""
    np.random.seed(id)
    color = np.random.randint(50, 255, size=3)
    return (int(color[0]), int(color[1]), int(color[2]))


def main():
    print("🚀 Initializing Drone MOT Benchmarking with Ground Truth...")

    drone = DroneController()
    vision = VisionSystem()
    tracker = DroneTracker()
    # BenchmarkRecorder will now automatically handle GT persistence filtering
    recorder = BenchmarkRecorder(config.ACTIVE_TRACKER) 

    time.sleep(2)
    drone.start_keyboard_control()

    print(f"\n📝 BENCHMARKING TRACKER: {config.ACTIVE_TRACKER}")
    print("🔴 Colors = Tracker IDs | 🟢 Green = Ground Truth (Persistent Only)")

    try:
        while drone.running:
            responses = drone.get_data()
            if responses is None:
                continue

            scene_response = responses[0]
            seg_response = responses[1]

            display_frame, results = vision.process_frame(scene_response)
            if display_frame is None or results is None:
                continue

            tracks, latency = tracker.update(results, display_frame.shape)

            # --- GROUND TRUTH EXTRACTION (No change here) ---
            h, w, _ = display_frame.shape

            # Decode segmentation image
            seg_id_map = decode_segmentation(seg_response)

            # Extract ALL current GT boxes (including potential noise/flicker)
            gt_boxes = extract_gt_boxes(
                seg_id_map,
                frame_width=w,
                frame_height=h,
                min_w=3,
                min_h=3,
                min_area=20,
                max_w_ratio=0.95,
                max_h_ratio=0.95,
                max_aspect_ratio=15.0,
                min_aspect_ratio=0.1,
                scale_with_distance=True,
                camera_pos=None,  
                object_depth_map=None
            )

            # -------------------------------------------------------------------
            # ✨ UPDATED GT LOGIC: Filter for persistence, log, and get persistent boxes
            # -------------------------------------------------------------------
            # This single call updates the recorder's state, logs the persistent boxes 
            # to the file, and returns only the boxes that met the threshold for drawing.
            persistent_gt_boxes = recorder.update_persistence_and_log_gt(
                vision.frame_count, 
                gt_boxes
            )
            
            # --- Draw GT boxes (ONLY persistent boxes are visualized) ---
            for gt_id, (cx, cy, bw, bh) in persistent_gt_boxes.items():
                if cx == -1:
                    continue  # skip off-screen objects
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"GT {gt_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                # NOTE: recorder.log_gt(vision.frame_count, gt_id, cx, cy, bw, bh) 
                # has been REMOVED as logging is now done inside the recorder's method.


            # --- TRACKER VISUALIZATION (No change) ---
            for cx, cy, w_box, h_box, conf, tid in tracks:
                color = get_color(tid)
                x1 = int(cx - w_box / 2)
                y1 = int(cy - h_box / 2)
                x2 = int(cx + w_box / 2)
                y2 = int(cy + h_box / 2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID {tid} ({conf:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + lw, y1), color, -1)
                cv2.putText(display_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                recorder.log(vision.frame_count, tid, cx, cy, w_box, h_box, conf, latency)

            # --- STATUS OVERLAY (No change) ---
            cv2.putText(display_frame, f"Tracker: {config.ACTIVE_TRACKER}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            cv2.putText(display_frame, f"Latency: {latency:.2f} ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

            cv2.imshow("Drone MOT Benchmark (GT + Tracking)", display_frame)

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                print("ESC pressed. Exiting...")
                break

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        recorder.close()
        drone.cleanup()
        vision.cleanup()
        cv2.destroyAllWindows()
        print("✅ Benchmark data saved successfully")


if __name__ == "__main__":
    main()
