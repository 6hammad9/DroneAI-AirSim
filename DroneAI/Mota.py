# run_mota_evaluation.py

import numpy as np
import motmetrics as mm
import pandas as pd
import os

def convert_center_to_top_left(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts bounding boxes from [CX, CY, W, H] to [X, Y, W, H].
    This is necessary because the logger saves center coordinates under 
    the 'X' and 'Y' column names.
    """
    # X_top_left = CX - W / 2
    df['X'] = df['X'] - (df['W'] / 2)
    # Y_top_left = CY - H / 2
    df['Y'] = df['Y'] - (df['H'] / 2)
    return df

def calculate_mot_scores_from_csv(log_path: str):
    """
    Loads ground truth and tracker output from CSV files, converts the format,
    and calculates MOTA and related scores.
    """
    gt_file = os.path.join(log_path, 'ground_truth.csv')
    pred_file = os.path.join(log_path, 'tracker_output.csv')
    
    if not os.path.exists(gt_file):
        print(f"ERROR: Ground truth file not found at {gt_file}")
        return
    if not os.path.exists(pred_file):
        print(f"ERROR: Tracker output file not found at {pred_file}")
        return

    print(f"Loading data from: {log_path}")
    
    # Load dataframes. Column names are now expected to be: FrameId, Id, X, Y, W, H
    try:
        # Use a list of dtypes to ensure coordinates and IDs are treated as numeric
        dtype_spec = {'FrameId': 'int32', 'Id': 'int32', 'X': 'float32', 'Y': 'float32', 'W': 'float32', 'H': 'float32'}
        df_gt = pd.read_csv(gt_file, dtype=dtype_spec, index_col=None)
        df_pred = pd.read_csv(pred_file, dtype=dtype_spec, index_col=None)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # --- CRITICAL STEP: CONVERSION ---
    # Convert from [CX, CY, W, H] to [TopLeft X, TopLeft Y, W, H] for accurate IoU
    df_gt = convert_center_to_top_left(df_gt.copy())
    df_pred = convert_center_to_top_left(df_pred.copy())
    
    # --- Setup MOT Accumulator ---
    acc = mm.MOTAccumulator(auto_id=True)
    
    # Get all unique frame IDs present in both files
    all_frames = sorted(list(set(df_gt['FrameId'].unique()) | set(df_pred['FrameId'].unique())))
    
    print(f"Processing {len(all_frames)} frames...")

    # --- Process Frame by Frame ---
    for frame_id in all_frames:
        
        # 1. Get GT for this frame
        gt_frame = df_gt[df_gt['FrameId'] == frame_id]
        gt_ids = gt_frame['Id'].to_numpy()
        gt_boxes = gt_frame[['X', 'Y', 'W', 'H']].to_numpy()
        
        # 2. Get Predictions for this frame
        pred_frame = df_pred[df_pred['FrameId'] == frame_id]
        pred_ids = pred_frame['Id'].to_numpy()
        pred_boxes = pred_frame[['X', 'Y', 'W', 'H']].to_numpy()

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            continue
        
        # 3. Calculate distance (IoU) matrix
        # A match requires IoU >= 0.5 (i.e., distance <= 0.5)
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

        # 4. Accumulate events
        acc.update(gt_ids, pred_ids, distances)

    # --- 5. Compute the metrics ---
    mh = mm.metrics.create()
    metrics = mm.metrics.motchallenge_metrics + ['num_frames']
    
    summary = mh.compute(
        acc, 
        metrics=metrics,
        name='BYTETrack_Evaluation'
    )
    
    return summary.T 


if __name__ == "__main__":
    
    # !!! IMPORTANT !!! 
    # REPLACE THIS PATH with the actual, newly generated log folder
    LOG_PATH = r"D:\drone_project\DroneAI\logs\BYTE_m3"
    
    print("Starting MOT Metrics Evaluation...")
    
    report = calculate_mot_scores_from_csv(LOG_PATH)
    
    if report is not None:
        print("\n" + "="*70)
        print("Multiple Object Tracking Metrics Report:")
        print("="*70)
        
        # --- FIX APPLIED HERE ---
        # Instead of using an unreliable motmetrics function, 
        # use the standard Pandas method for rendering the DataFrame.
        strsummary = report.to_string() 
        # --- END FIX ---
        
        print(strsummary)

        # The rest of the file saving remains the same
        report_file = os.path.join(LOG_PATH, 'mota_report.txt')
        with open(report_file, 'w') as f:
            f.write(strsummary)
        print(f"\nReport saved to: {report_file}")