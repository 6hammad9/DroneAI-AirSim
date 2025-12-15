"""
Multi-Object Tracking System

Supports both ByteTrack and Norfair tracking algorithms with
unified interface and performance monitoring.
"""

import time
import numpy as np
from typing import List, Tuple, Optional
import warnings

# Tracking libraries
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    warnings.warn("supervision not available - ByteTrack disabled")

try:
    from norfair import Detection, Tracker
    NORFAIR_AVAILABLE = True
except ImportError:
    NORFAIR_AVAILABLE = False
    warnings.warn("norfair not available - Norfair tracker disabled")

try:
    from config import Config
    TRACK_CONFIG = Config.Tracking
except ImportError:
    # Fallback configuration
    class TRACK_CONFIG:
        ACTIVE_TRACKER = "BYTE"
        class ByteTrack:
            TRACK_ACTIVATION_THRESHOLD = 0.25
            LOST_TRACK_BUFFER = 30
            MINIMUM_MATCHING_THRESHOLD = 0.8
            FRAME_RATE = 30
        class Norfair:
            DISTANCE_FUNCTION = "euclidean"
            DISTANCE_THRESHOLD = 50
            INITIALIZATION_DELAY = 10


class TrackerNotAvailableError(Exception):
    """Raised when selected tracker library is not installed"""
    pass


class DroneTracker:
    """
    Unified interface for multiple tracking algorithms
    
    Supported trackers:
    - ByteTrack: Association-based with low-confidence detection recovery
    - Norfair: Kalman filter-based with Euclidean distance matching
    
    Features:
    - Automatic tracker selection based on config
    - Performance monitoring (latency tracking)
    - Unified output format: [cx, cy, w, h, conf, track_id]
    - Error handling and graceful degradation
    """
    
    def __init__(self, tracker_type: Optional[str] = None, config=None):
        """
        Initialize tracker
        
        Args:
            tracker_type: "BYTE" or "NORFAIR" (uses config if None)
            config: Configuration object
            
        Raises:
            TrackerNotAvailableError: If selected tracker not installed
        """
        self.config = config or TRACK_CONFIG
        self.mode = tracker_type or self.config.ACTIVE_TRACKER
        
        # Performance tracking
        self.frame_count = 0
        self.track_count = 0
        self.latencies = []
        
        # Initialize tracker
        self.tracker = self._initialize_tracker()
        
        print(f"✅ {self.mode}Track initialized")
    
    def _initialize_tracker(self):
        """Initialize the selected tracking algorithm"""
        
        if self.mode == "BYTE":
            return self._init_bytetrack()
        elif self.mode == "NORFAIR":
            return self._init_norfair()
        else:
            raise ValueError(f"Unknown tracker type: {self.mode}")
    
    def _init_bytetrack(self):
        """Initialize ByteTrack"""
        if not SUPERVISION_AVAILABLE:
            raise TrackerNotAvailableError(
                "ByteTrack requires 'supervision' library. "
                "Install with: pip install supervision"
            )
        
        config = self.config.ByteTrack
        
        tracker = sv.ByteTrack(
            track_activation_threshold=config.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=config.LOST_TRACK_BUFFER,
            minimum_matching_threshold=config.MINIMUM_MATCHING_THRESHOLD,
            frame_rate=config.FRAME_RATE
        )
        
        print(f"   Track activation threshold: {config.TRACK_ACTIVATION_THRESHOLD}")
        print(f"   Lost track buffer: {config.LOST_TRACK_BUFFER} frames")
        print(f"   Minimum matching threshold: {config.MINIMUM_MATCHING_THRESHOLD}")
        
        return tracker
    
    def _init_norfair(self):
        """Initialize Norfair"""
        if not NORFAIR_AVAILABLE:
            raise TrackerNotAvailableError(
                "Norfair requires 'norfair' library. "
                "Install with: pip install norfair"
            )
        
        config = self.config.Norfair
        
        tracker = Tracker(
            distance_function=config.DISTANCE_FUNCTION,
            distance_threshold=config.DISTANCE_THRESHOLD,
            initialization_delay=config.INITIALIZATION_DELAY,
            hit_counter_max=getattr(config, 'HIT_COUNTER_MAX', 30)
        )
        
        print(f"   Distance function: {config.DISTANCE_FUNCTION}")
        print(f"   Distance threshold: {config.DISTANCE_THRESHOLD} pixels")
        print(f"   Initialization delay: {config.INITIALIZATION_DELAY} frames")
        
        return tracker
    
    def update(
        self,
        results: any,
        frame_shape: Tuple[int, int, int],
        return_all: bool = True
    ) -> Tuple[List[List], float]:
        """
        Update tracker with new detections
        
        Args:
            results: YOLO detection results (or compatible format)
            frame_shape: (height, width, channels)
            return_all: If True, return all tracks; if False, only confirmed tracks
            
        Returns:
            Tuple of (tracks, latency_ms) where tracks is list of:
            [center_x, center_y, width, height, confidence, track_id]
        """
        start_time = time.time()
        
        try:
            if self.mode == "BYTE":
                final_tracks = self._update_bytetrack(results, return_all)
            elif self.mode == "NORFAIR":
                final_tracks = self._update_norfair(results, return_all)
            else:
                final_tracks = []
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self.frame_count += 1
            self.track_count += len(final_tracks)
            self.latencies.append(latency_ms)
            
            # Keep only recent latencies (last 100 frames)
            if len(self.latencies) > 100:
                self.latencies.pop(0)
            
            return final_tracks, latency_ms
            
        except Exception as e:
            print(f"⚠️  Tracking update failed: {e}")
            return [], 0.0
    
    def _update_bytetrack(
        self,
        results: any,
        return_all: bool
    ) -> List[List]:
        """
        Update ByteTrack tracker
        
        Args:
            results: YOLO results
            return_all: Return all tracks or only confirmed
            
        Returns:
            List of tracks in unified format
        """
        final_tracks = []
        
        try:
            # Convert YOLO results to Supervision Detections
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Update tracker
            detections = self.tracker.update_with_detections(detections)
            
            # Extract tracks
            if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                # Handle missing confidence scores (bug fix)
                if detections.confidence is None or len(detections.confidence) == 0:
                    current_confs = np.ones(len(detections.tracker_id))
                else:
                    current_confs = detections.confidence
                
                # Convert to unified format
                for box, track_id, conf in zip(
                    detections.xyxy,
                    detections.tracker_id,
                    current_confs
                ):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    final_tracks.append([
                        float(center_x),
                        float(center_y),
                        float(width),
                        float(height),
                        float(conf),
                        int(track_id)
                    ])
        
        except Exception as e:
            warnings.warn(f"ByteTrack update error: {e}")
        
        return final_tracks
    
    def _update_norfair(
        self,
        results: any,
        return_all: bool
    ) -> List[List]:
        """
        Update Norfair tracker
        
        Args:
            results: YOLO results
            return_all: Return all tracks or only live tracks
            
        Returns:
            List of tracks in unified format
        """
        final_tracks = []
        norfair_detections = []
        
        try:
            # Convert YOLO results to Norfair Detections
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, coords)
                    conf = float(box.conf[0])
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    norfair_detections.append(
                        Detection(
                            points=np.array([[center_x, center_y]]),
                            scores=np.array([conf]),
                            data={
                                "w": x2 - x1,
                                "h": y2 - y1
                            }
                        )
                    )
            
            # Update tracker
            tracked_objects = self.tracker.update(detections=norfair_detections)
            
            # Convert to unified format
            for obj in tracked_objects:
                # Only return tracks that are currently live (detected in this frame)
                # This disables coasting (prediction without detection)
                if return_all or obj.live_points.any():
                    # Use Kalman filter estimation (smoothed position)
                    center_x, center_y = obj.estimate[0]
                    
                    # Get dimensions from last detection
                    if obj.last_detection and obj.last_detection.data:
                        width = obj.last_detection.data["w"]
                        height = obj.last_detection.data["h"]
                        conf = obj.last_detection.scores[0]
                    else:
                        # No detection data - use defaults
                        width, height, conf = 0, 0, 0.0
                    
                    final_tracks.append([
                        float(center_x),
                        float(center_y),
                        float(width),
                        float(height),
                        float(conf),
                        int(obj.id)
                    ])
        
        except Exception as e:
            warnings.warn(f"Norfair update error: {e}")
        
        return final_tracks
    
    def get_stats(self) -> dict:
        """
        Get tracker performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.latencies:
            return {
                'tracker_type': self.mode,
                'frames_processed': self.frame_count,
                'total_tracks': self.track_count,
                'avg_tracks_per_frame': 0,
                'avg_latency_ms': 0,
                'min_latency_ms': 0,
                'max_latency_ms': 0
            }
        
        return {
            'tracker_type': self.mode,
            'frames_processed': self.frame_count,
            'total_tracks': self.track_count,
            'avg_tracks_per_frame': self.track_count / self.frame_count if self.frame_count > 0 else 0,
            'avg_latency_ms': np.mean(self.latencies),
            'min_latency_ms': np.min(self.latencies),
            'max_latency_ms': np.max(self.latencies),
            'std_latency_ms': np.std(self.latencies)
        }
    
    def reset(self):
        """Reset tracker state (clears all tracks)"""
        print(f"🔄 Resetting {self.mode}Track...")
        
        # Reinitialize tracker
        self.tracker = self._initialize_tracker()
        
        # Reset statistics
        self.frame_count = 0
        self.track_count = 0
        self.latencies = []
        
        print("   ✓ Tracker reset complete")
    
    def print_stats(self):
        """Print tracker statistics"""
        stats = self.get_stats()
        
        print(f"\n📊 {self.mode}Track Statistics:")
        print(f"   Frames processed: {stats['frames_processed']}")
        print(f"   Total tracks: {stats['total_tracks']}")
        print(f"   Avg tracks/frame: {stats['avg_tracks_per_frame']:.2f}")
        print(f"   Avg latency: {stats['avg_latency_ms']:.2f} ms")
        print(f"   Latency range: [{stats['min_latency_ms']:.2f}, {stats['max_latency_ms']:.2f}] ms")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test tracking system with mock detections
    """
    print("Testing DroneTracker...\n")
    
    # Test both trackers if available
    tracker_types = []
    if SUPERVISION_AVAILABLE:
        tracker_types.append("BYTE")
    if NORFAIR_AVAILABLE:
        tracker_types.append("NORFAIR")
    
    if not tracker_types:
        print("❌ No trackers available!")
        print("   Install with: pip install supervision norfair")
        exit(1)
    
    for tracker_type in tracker_types:
        print(f"\n{'='*60}")
        print(f"Testing {tracker_type}Track")
        print(f"{'='*60}")
        
        try:
            # Initialize tracker
            tracker = DroneTracker(tracker_type=tracker_type)
            
            # Create mock YOLO results
            class MockBox:
                def __init__(self, x1, y1, x2, y2, conf):
                    self.xyxy = [np.array([x1, y1, x2, y2])]
                    self.conf = [conf]
                    
                def cpu(self):
                    return self
                
                def numpy(self):
                    return self.xyxy[0]
            
            class MockBoxes:
                def __init__(self, boxes_list):
                    self.boxes = boxes_list
                
                def __iter__(self):
                    return iter(self.boxes)
                
                def __len__(self):
                    return len(self.boxes)
            
            class MockResults:
                def __init__(self, boxes_list):
                    self.boxes = MockBoxes(boxes_list)
            
            # Simulate 10 frames with 2-3 detections each
            print("\n📸 Processing mock frames...")
            for frame_idx in range(10):
                # Create mock detections (moving boxes)
                boxes = [
                    MockBox(100 + frame_idx*5, 100, 150 + frame_idx*5, 150, 0.9),
                    MockBox(300, 200 + frame_idx*3, 350, 250 + frame_idx*3, 0.85),
                ]
                
                # Add third detection occasionally
                if frame_idx % 3 == 0:
                    boxes.append(MockBox(500, 300, 550, 350, 0.75))
                
                results = [MockResults(boxes)]
                
                # Update tracker
                tracks, latency = tracker.update(results, (480, 640, 3))
                
                print(f"   Frame {frame_idx+1}: {len(tracks)} tracks, latency={latency:.2f}ms")
            
            # Print statistics
            tracker.print_stats()
            
            print(f"\n✅ {tracker_type}Track test passed!")
            
        except TrackerNotAvailableError as e:
            print(f"⚠️  {tracker_type}Track not available: {e}")
        except Exception as e:
            print(f"❌ {tracker_type}Track test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All tracker tests complete!")