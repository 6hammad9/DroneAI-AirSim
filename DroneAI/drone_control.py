import airsim
import threading
import time
from pynput import keyboard
import config

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client_lock = threading.Lock()
        self.key_lock = threading.Lock()
        self.pressed_keys = set()
        self.running = True
        
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("✅ Connected to AirSim")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            exit()

    def safe_call(self, func, *args, **kwargs):
        with self.client_lock:
            return func(*args, **kwargs)

    def start_keyboard_control(self):
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.vel_thread = threading.Thread(target=self._update_velocity, daemon=True)
        self.vel_thread.start()

    def _update_velocity(self):
        while self.running:
            with self.key_lock:
                vx, vy, vz, yaw_rate = 0, 0, 0, 0
                
                if keyboard.Key.up in self.pressed_keys: vx = config.MOVE_SPEED
                elif keyboard.Key.down in self.pressed_keys: vx = -config.MOVE_SPEED
                
                if keyboard.Key.left in self.pressed_keys: yaw_rate = -config.YAW_SPEED
                elif keyboard.Key.right in self.pressed_keys: yaw_rate = config.YAW_SPEED

                if self._is_char_pressed('a'): vy = -config.MOVE_SPEED
                elif self._is_char_pressed('d'): vy = config.MOVE_SPEED
                
                if keyboard.Key.page_up in self.pressed_keys: vz = -config.ALT_SPEED
                elif keyboard.Key.page_down in self.pressed_keys: vz = config.ALT_SPEED

            self.safe_call(
                self.client.moveByVelocityBodyFrameAsync, 
                vx, vy, vz, 0.1, 
                airsim.DrivetrainType.MaxDegreeOfFreedom, 
                airsim.YawMode(True, yaw_rate)
            )
            time.sleep(0.05)

    def _is_char_pressed(self, char):
        return hasattr(keyboard.KeyCode.from_char(char), 'vk') and keyboard.KeyCode.from_char(char) in self.pressed_keys

    def _on_press(self, key):
        try:
            k = key.char
            if k == 'q': self.running = False
            elif k in ['t', 'l', 'h']:
                threading.Thread(target=self._run_action, args=(k,), daemon=True).start()
                if k == 'h':
                    with self.key_lock: self.pressed_keys.clear()
            with self.key_lock: self.pressed_keys.add(key)
        except AttributeError:
            with self.key_lock: self.pressed_keys.add(key)

    def _on_release(self, key):
        with self.key_lock: self.pressed_keys.discard(key)

    def _run_action(self, action):
        if action == 't': self.safe_call(self.client.takeoffAsync).join()
        elif action == 'l': self.safe_call(self.client.landAsync).join()
        elif action == 'h': self.safe_call(self.client.hoverAsync).join()

    def get_data(self):
        with self.client_lock:
            responses = self.client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest(
        "0", 
        airsim.ImageType.Segmentation, 
        False,         # pixels_as_float = False
        False          # compress = False (CRITICAL!)
    )
          ])
        return responses if responses else None


    def cleanup(self):
        self.running = False
        time.sleep(0.2)
        if hasattr(self, 'listener'): self.listener.stop()
        try:
            self.safe_call(self.client.armDisarm, False)
            self.safe_call(self.client.enableApiControl, False)
        except: pass