# config.py
import os

# --- DRONE SETTINGS ---
MOVE_SPEED = 3    
ALT_SPEED = 2    
YAW_SPEED = 50   

# --- MODEL SETTINGS ---
MODEL_PATH = "best.pt"   # Your trained model
CONF_THRESHOLD = 0.25 
YOLO_CLASSES = [0]       # Class 0 = Person

# --- TRACKER SETTINGS ---
# CHANGE THIS for your paper experiments: "BYTE" or "NORFAIR"
ACTIVE_TRACKER = "BYTE"  

# Set to FALSE for MOT paper (we want to see how it handles crowds)
ONLY_TRACK_CENTER_OBJECT = False 

# --- SAVE SETTINGS ---
SAVE_DIR = "benchmarking"
SAVE_COOLDOWN = 1.0      
os.makedirs(SAVE_DIR, exist_ok=True)