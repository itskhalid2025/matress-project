import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# ===========================
# Load model
# ===========================

interpreter = tflite.Interpreter(
    model_path="Mattress_Texture_Model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===========================
# Classes
# ===========================

class_names = [
    "Dual harmony",
    "Gravite",
    "Maxi plush",
    "Maxi pro",
    "Memorise",
    "Ortholex",
    "Purity plus",
    "Velvet"
]

# ===========================
# Camera
# ===========================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    image = cv2.resize(frame, (224,224))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32)

    image = (image / 127.5) - 1

    image = np.expand_dims(image, axis=0)

    start = time.time()

    interpreter.set_tensor(
        input_details[0]['index'],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]['index']
    )[0]

    prediction = np.argmax(output)

    confidence = output[prediction]*100

    end = time.time()

    fps = 1/(end-start)

    text = f"{class_names[prediction]}  {confidence:.1f}%"

    cv2.putText(
        frame,
        text,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    cv2.imshow("Mattress Classifier", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()


#sudo apt install python3-picamera2
#sudo apt update

#pip3 install tflite-runtime opencv-python numpy