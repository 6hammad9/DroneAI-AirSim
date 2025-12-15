# ====================================================================
# mot_evaluator.py - FINAL STRUCTURALLY AND LOGICALLY CORRECT VERSION
# ====================================================================

import pandas as pd
import motmetrics as mm
import glob
import numpy as np
import os

# --- CONFIGURATION ---
BYTE_TRACK_FILE_PATTERN = "RESULTS_BYTE_*.csv"
NORFAIR_TRACK_FILE_PATTERN = "RESULTS_NORFAIR_*.csv"
TARGET_SCENARIO = "Rapid Yaw Maneuver" 


# ==========================================================
# 1. LOAD AND PREPARE DATA (MUST BE DEFINED FIRST)
# ==========================================================
def load_and_prepare_data(file_pattern, is_tracker_output=False):
    """Loads a tracking CSV, checks for normalized coordinates, and prepares data."""
    
    try:
        filename = max(glob.glob(file_pattern), key=os.path.getctime)
    except ValueError:
        print(f"❌ ERROR: No file found matching pattern: {file_pattern}")
        return None, None
        
    df = pd.read_csv(filename)
    
    # ⚠️ FIX: COORDINATE ALIGNMENT HEURISTICS (Scales normalized data to 1280x720)
    if df['Center_X'].max() < 2.0 or df['Width'].max() < 2.0:
        WIDTH_SCALE = 1280 
        HEIGHT_SCALE = 720
        
        print(f"⚠️ Detected normalized coordinates in {os.path.basename(filename)}. Scaling to {WIDTH_SCALE}x{HEIGHT_SCALE}.")
        
        df['Center_X'] *= WIDTH_SCALE
        df['Width'] *= WIDTH_SCALE
        df['Center_Y'] *= HEIGHT_SCALE
        df['Height'] *= HEIGHT_SCALE
        
    # Convert Center_X/Y to Top-left X/Y
    df['X'] = df['Center_X'] - df['Width'] / 2
    df['Y'] = df['Center_Y'] - df['Height'] / 2

    df = df.rename(columns={'Frame': 'FrameId', 'Track_ID': 'ObjectId'})

    # Calculate FPS
    avg_fps = None
    if is_tracker_output and 'Latency_ms' in df.columns:
        avg_latency_ms = df['Latency_ms'].mean()
        avg_fps = 1000 / avg_latency_ms if avg_latency_ms > 0 else np.nan

    final_df = df[['FrameId', 'ObjectId', 'X', 'Y', 'Width', 'Height']].copy()

    print(f"✅ Loaded data from {filename}. Total frames: {final_df['FrameId'].max()}")
    return final_df, avg_fps


# ==========================================================
# 2. EVALUATE TRACKER (EUCLIDEAN DISTANCE)
# ==========================================================
def evaluate_tracker(gt_df, tracker_df, tracker_name, avg_fps):
    """Calculates MOT metrics using Euclidean Distance (L2 Norm) for stable matching."""
    
    acc = mm.MOTAccumulator(auto_id=True)
    common_frames = sorted(set(gt_df['FrameId']).intersection(set(tracker_df['FrameId'])))
    
    if not common_frames:
        print("❌ CRITICAL ERROR: No overlapping frames found. Check your logging file names!")
        return pd.Series({'mota': np.nan, 'motp': np.nan, 'Avg_FPS': avg_fps})

    
    print("✅ Switching matching criterion to Euclidean Distance (L2 Norm).")
    
    # Define a pixel threshold for association (e.g., 50 pixels)
    PIXEL_THRESHOLD = 50.0 
    DISTANCE_THRESHOLD_SQUARED = PIXEL_THRESHOLD ** 2 


    for frameid in common_frames:
        
        gt_frame_df = gt_df.loc[gt_df['FrameId'] == frameid]
        tracker_frame_df = tracker_df.loc[tracker_df['FrameId'] == frameid]

        # Extract only Center_X and Center_Y (X, Y are top-left, but we treat them as centers for L2 Norm)
        gt_coords = gt_frame_df[['X', 'Y']].values.astype(np.float64)
        tracker_coords = tracker_frame_df[['X', 'Y']].values.astype(np.float64)
        
        gt_ids = gt_frame_df['ObjectId'].values
        tracker_ids = tracker_frame_df['ObjectId'].values

        # Use L2 Norm (Euclidean Distance Squared) for matching
        distances = mm.distances.norm2squared_matrix(
            gt_coords, 
            tracker_coords, 
            max_d2=DISTANCE_THRESHOLD_SQUARED
        ) 
        
        acc.update(gt_ids, tracker_ids, distances)

    # --- Dynamic Metric Name Detection (Final check against Assertion Error) ---
    mh = mm.metrics.create()
    available_metrics = set(mh.list_metrics())
    
    metrics_to_compute = ['mota', 'motp', 'num_frames', 'idsw', 'num_switches', 'ids']
    metrics_to_compute = [m for m in metrics_to_compute if m in available_metrics]

    print(f"📊 Forcing metrics computation using: {metrics_to_compute}")

    summary = mh.compute(
        acc, 
        metrics=metrics_to_compute, 
        name=tracker_name
    )
    
    summary['Avg_FPS'] = avg_fps
    summary['IDS_Metric_Name'] = next((m for m in ['idsw', 'num_switches', 'ids'] if m in summary.index), 'N/A')
    
    return summary


# ==========================================================
# 3. MAIN COMPARISON FUNCTION (Definition)
# ==========================================================
def main():
    print(f"\n🚀 Starting MOT Benchmarking for Scenario: {TARGET_SCENARIO}\n")

    # --- Load Data ---
    # The NameError is fixed because load_and_prepare_data is defined above.
    gt_data, byte_fps = load_and_prepare_data(BYTE_TRACK_FILE_PATTERN, is_tracker_output=True)
    if gt_data is None: return

    tracker_data, norfair_fps = load_and_prepare_data(NORFAIR_TRACK_FILE_PATTERN, is_tracker_output=True)
    if tracker_data is None: return

    # --- Evaluate ---
    norfair_summary = evaluate_tracker(gt_data, tracker_data, "Norfair_vs_ByteTrack_GT", norfair_fps)

    print("\n" + "=" * 55)
    print(f"🏁 BENCHMARK RESULTS ({TARGET_SCENARIO})")
    print(" (ByteTrack's output used as Pseudo-Ground Truth)")
    print("=" * 55)

    results_list = []

    # --- Helper function for robust IDS extraction ---
    def get_ids_value(summary_series):
        # Checks for the three possible IDS keys in order of likelihood
        return summary_series.get('idsw', norfair_summary.get('num_switches', norfair_summary.get('ids', np.nan)))

    # --- Add ByteTrack baseline ---
    results_list.append({
        'Tracker': 'ByteTrack (Baseline)',
        'MOTA': np.nan,
        'ID Switches (IDS)': np.nan,
        'MOTP': np.nan,
        'Avg_FPS': byte_fps
    })

    # --- Extract Norfair row ---
    norfair_row = norfair_summary
    
    results_list.append({
        'Tracker': 'Norfair (vs ByteTrack GT)',
        'MOTA': norfair_row.get('mota', np.nan),
        'ID Switches (IDS)': get_ids_value(norfair_row),
        'MOTP': norfair_row.get('motp', np.nan),
        'Avg_FPS': norfair_row.get('Avg_FPS', np.nan)
    })

    # --- Final Formatting and Output ---
    final_comparison_df = pd.DataFrame(results_list)
    final_comparison_df = final_comparison_df.set_index('Tracker')

    print(final_comparison_df.to_string(float_format="{:.4f}".format))

    output_filename = f"MOT_Results_{TARGET_SCENARIO.replace(' ', '_')}.csv"
    final_comparison_df.to_csv(output_filename)
    print(f"\n💾 Results saved to: {os.path.abspath(output_filename)}")


# ==========================================================
# 4. ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    main()