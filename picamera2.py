import os
import sys
import platform

# Identify if we are in an environment that should use the mock or try to load the real picamera2.
# On Windows, we always use the mock because the real picamera2 is not supported.
# On Linux (like Raspberry Pi OS), we try to load the real picamera2 from system packages.
use_mock = True

if platform.system() != "Windows":
    # Try importing the real picamera2 from system packages (excluding the local directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = sys.path.copy()
    
    # Remove local directories from search path temporarily
    sys.path = [p for p in sys.path if p not in (current_dir, '', '.')]
    
    # Temporarily remove ourselves from sys.modules so the import isn't cached
    self_module = sys.modules.pop('picamera2', None)
    
    try:
        import picamera2 as real_picamera2
        # Expose all attributes of real_picamera2 in this module
        globals().update({k: v for k, v in real_picamera2.__dict__.items() if not k.startswith('__')})
        use_mock = False
    except ImportError:
        pass
    finally:
        # Restore sys.path and sys.modules
        sys.path = original_path
        if self_module:
            sys.modules['picamera2'] = self_module

if use_mock:
    # --- MOCK IMPLEMENTATION FOR WINDOWS / DEVELOPMENT ---
    import numpy as np

    class Picamera2:
        def __init__(self, *args, **kwargs):
            print("[Mock Picamera2] Initialized mock camera")
            self.running = False

        def create_video_configuration(self, *args, **kwargs):
            print(f"[Mock Picamera2] create_video_configuration with args: {args}, kwargs: {kwargs}")
            return {"mock_config": True}

        def configure(self, config):
            print(f"[Mock Picamera2] configure with config: {config}")

        def start(self):
            print("[Mock Picamera2] start")
            self.running = True

        def stop(self):
            print("[Mock Picamera2] stop")
            self.running = False

        def close(self):
            print("[Mock Picamera2] close")

        def capture_array(self):
            # Return a dummy image frame (e.g., 640x480x3 black frame with some text or patterns)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add a subtle color pattern so it's not completely black
            frame[:, :] = [40, 40, 40]
            return frame
