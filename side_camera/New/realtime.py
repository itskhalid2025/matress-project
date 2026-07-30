from flask import Flask, Response
import cv2
import numpy as np
import time

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

app = Flask(__name__)

# Load Model
interpreter = tflite.Interpreter(model_path="Mattress_Texture_Model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
    "Dual harmony", "Gravite", "Maxi plush", "Maxi pro",
    "Memorise", "Ortholex", "Purity plus", "Velvet"
]

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        image = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = (image / 127.5) - 1
        image = np.expand_dims(image, axis=0)

        # Inference
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        end = time.time()

        prediction = np.argmax(output)
        confidence = output[prediction] * 100
        fps = 1 / (end - start) if (end - start) > 0 else 0

        # Draw overlays on the frame
        text = f"{class_names[prediction]}  {confidence:.1f}%"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Stream frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Listens on all local network interfaces on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)