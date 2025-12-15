import cv2
import time
import numpy as np
import threading
import queue
from ultralytics import YOLO
import config

class VisionSystem:
    def __init__(self):
        try:
            self.model = YOLO(config.MODEL_PATH)
            print("✅ YOLO model loaded")
        except Exception as e:
            print(f"❌ YOLO Error: {e}")
            self.model = None

        self.last_save_time = 0
        self.frame_count = 0
        
        self.save_queue = queue.Queue()
        self.writer_running = True
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def process_frame(self, response):
        """Merges Image and YOLO processing"""
        if not response: return None, None

        # 1. Use frombuffer instead of fromstring (faster & modern)
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
        
        # 2. CRITICAL FIX: .copy() makes the array writeable.
        # Without this, OpenCV crashes because AirSim buffers are read-only.
        frame = img1d.reshape(response.height, response.width, 3).copy()
        
        self.frame_count += 1
        
        # --- RUN YOLO ---
        if self.model:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb_frame, verbose=False, classes=config.YOLO_CLASSES, conf=config.CONF_THRESHOLD)
            
            # Draw simple green boxes for raw detection verification
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            self._check_and_queue_save(results, frame)
            
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Return Frame (for display) AND Results (for Tracker)
            return frame, results
        
        return frame, None

    def _check_and_queue_save(self, results, frame):
        has_low_conf = False
        min_conf = 1.0

        for box in results[0].boxes:
            conf = float(box.conf[0])
            min_conf = min(min_conf, conf)
            if conf < config.CONF_THRESHOLD:
                has_low_conf = True
        
        current_time = time.time()
        if has_low_conf and (current_time - self.last_save_time) > config.SAVE_COOLDOWN:
            timestamp = int(current_time * 1000)
            filename = f"{config.SAVE_DIR}/bench_{timestamp}_conf{min_conf:.2f}.jpg"
            self.save_queue.put((filename, frame.copy()))
            self.last_save_time = current_time

    def _writer_worker(self):
        while self.writer_running:
            try:
                filename, img = self.save_queue.get(timeout=1)
                cv2.imwrite(filename, img)
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def cleanup(self):
        self.writer_running = False
        if self.writer_thread.is_alive():
            self.writer_thread.join()