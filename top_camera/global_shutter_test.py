import cv2
import socket
from flask import Flask, Response

app = Flask(__name__)

def generate_frames():
    # Connect to the camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Set to Global Shutter resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1456)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1088)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break
            
        # ==========================================
        # YOUR ULTRALYTICS/PYTORCH CODE WILL GO HERE
        # e.g., results = model(frame)
        # ==========================================

        # Encode the frame as a JPEG for the web browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the properly formatted MJPEG boundary
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == "__main__":
    ip_addr = get_ip()
    print("\n" + "="*50)
    print(" PYTHON 3.11 OPENCV STREAM ACTIVE!")
    print(f" Open this URL in your web browser: http://{ip_addr}:5000/")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)