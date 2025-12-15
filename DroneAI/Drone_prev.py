import os

import airsim

import cv2

import numpy as np

import time

from ultralytics import YOLO

from pynput import keyboard

import threading



# ---------- CONFIG ----------

MODEL_PATH = "best.pt"

CONF_THRESHOLD = 0.3

SAVE_DIR = "confused"

os.makedirs(SAVE_DIR, exist_ok=True)



model = YOLO(MODEL_PATH)

print("✅ YOLOv11 model loaded")



# ---------- CONNECT TO AIRSIM ----------

client = airsim.MultirotorClient()

client.confirmConnection()

client.enableApiControl(True)

client.armDisarm(True)

print("✅ Connected to AirSim")



# ---------- THREAD-SAFE CLIENT WRAPPER ----------

client_lock = threading.Lock()



def safe_client_call(func, *args, **kwargs):

    """Thread-safe wrapper for AirSim client calls."""

    with client_lock:

        return func(*args, **kwargs)



# ---------- GLOBAL FLAGS ----------

running = True

move_speed = 3  # m/s

alt_speed = 2   # m/s

last_save_time = 0

SAVE_COOLDOWN = 2  # seconds between saves



# Track currently pressed keys

pressed_keys = set()

key_lock = threading.Lock()



# ---------- MOVEMENT CONTROL ----------

def update_velocity():

    """Continuously update drone velocity based on pressed keys."""

    global running

   

    while running:

        with key_lock:

            vx, vy, vz = 0, 0, 0

           

            if keyboard.Key.up in pressed_keys:

                vx = move_speed

            elif keyboard.Key.down in pressed_keys:

                vx = -move_speed

               

            if keyboard.Key.left in pressed_keys:

                vy = -move_speed

            elif keyboard.Key.right in pressed_keys:

                vy = move_speed

               

            if keyboard.Key.page_up in pressed_keys:

                vz = -alt_speed

            elif keyboard.Key.page_down in pressed_keys:

                vz = alt_speed

       

        # Thread-safe velocity command

        safe_client_call(client.moveByVelocityAsync, vx, vy, vz, 1)

        time.sleep(0.1)  # Update 10 times per second



# Start velocity control thread

velocity_thread = threading.Thread(target=update_velocity, daemon=True)

velocity_thread.start()



# ---------- KEYBOARD CALLBACKS ----------

def on_press(key):

    global running

   

    try:

        if key.char == 'q':  # Quit

            running = False

        elif key.char == 't':  # Takeoff

            print("🚁 Taking off...")

            safe_client_call(client.takeoffAsync).join()

            print("✅ Takeoff complete")

        elif key.char == 'l':  # Land

            print("🛬 Landing...")

            safe_client_call(client.landAsync).join()

            print("✅ Landed")

        elif key.char == 'h':  # Hover (stop)

            with key_lock:

                pressed_keys.clear()

            safe_client_call(client.hoverAsync).join()

            print("⏸️  Hovering")

    except AttributeError:

        # Arrow keys and special keys

        with key_lock:

            pressed_keys.add(key)



def on_release(key):

    """Remove key from pressed set when released."""

    with key_lock:

        pressed_keys.discard(key)



# ---------- START KEYBOARD LISTENER ----------

listener = keyboard.Listener(on_press=on_press, on_release=on_release)

listener.start()



print("\n🎮 CONTROLS:")

print("  T: Takeoff")

print("  L: Land")

print("  H: Hover (stop)")

print("  Arrow Keys: Move (Forward/Back/Left/Right)")

print("  Page Up/Down: Altitude")

print("  Q: Quit")

print("  ESC: Emergency stop\n")



# ---------- LIVE FEED WITH YOLO ----------

frame_count = 0



try:

    while running:

        # Thread-safe image retrieval

        response = safe_client_call(client.simGetImage, "0", airsim.ImageType.Scene)

       

        if response is None or len(response) == 0:

            time.sleep(0.01)

            continue



        img1d = np.frombuffer(response, dtype=np.uint8)

        frame = cv2.imdecode(img1d, cv2.IMREAD_COLOR)

        img = safe_client_call(client.simGetImage, "0", airsim.ImageType.Scene)

        print("Image size:", np.frombuffer(img, dtype=np.uint8).shape)



        if frame is None:

            continue



        frame_count += 1



        # YOLO detection

        results = model(frame, verbose=False)

        annotated_frame = results[0].plot()



        # Check for low-confidence detections

        has_low_conf = False

        min_conf = 1.0

       

        for box in results[0].boxes:

            conf = float(box.conf[0])

            min_conf = min(min_conf, conf)

            if conf < CONF_THRESHOLD:

                has_low_conf = True



        # Save low-confidence frames (with cooldown)

        current_time = time.time()

        if has_low_conf and (current_time - last_save_time) > SAVE_COOLDOWN:

            timestamp = int(current_time * 1000)

            filename = f"{SAVE_DIR}/conf{min_conf:.2f}_{timestamp}.jpg"

            cv2.imwrite(filename, frame)

            last_save_time = current_time

            print(f"💾 Saved low-confidence frame: {filename}")



        # Add info overlay

        info_text = f"Frame: {frame_count} | Detections: {len(results[0].boxes)}"

        if len(results[0].boxes) > 0:

            info_text += f" | Min Conf: {min_conf:.2f}"

       

        cv2.putText(annotated_frame, info_text, (10, 30),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



        # Show feed

        # display_frame = cv2.resize(annotated_frame, (1280, 720))

        # cv2.namedWindow("Drone YOLO Feed", cv2.WINDOW_NORMAL)

        cv2.imshow("Drone YOLO Feed", annotated_frame)

       



        # ESC to quit

        if cv2.waitKey(1) & 0xFF == 27:

            running = False

            break



except KeyboardInterrupt:

    print("\n⚠️  Interrupted by user")

    running = False

except Exception as e:

    print(f"\n❌ Error: {e}")

    running = False



# ---------- CLEANUP ----------

print("\n🧹 Cleaning up...")

running = False

time.sleep(0.2)  # Let velocity thread finish

listener.stop()



# Emergency hover before landing

try:

    safe_client_call(client.hoverAsync).join()

    time.sleep(0.5)

    safe_client_call(client.landAsync).join()

    safe_client_call(client.armDisarm, False)

    safe_client_call(client.enableApiControl, False)

except:

    pass



cv2.destroyAllWindows()

print("✅ Mission complete!")

