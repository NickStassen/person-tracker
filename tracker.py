import cv2
from flask import Flask, Response
import threading
import time

MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
CONFIG_PATH = "MobileNetSSD_deploy.prototxt"
CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

CONFIDENCE_THRESHOLD = 0.5
DETECTION_INTERVAL = 15

net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
tracker = None
bbox_lock = threading.Lock()
last_bbox = None
frame_count = 0

def detect_and_track():
    global tracker, frame_count, last_bbox

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            if frame_count % DETECTION_INTERVAL == 0 or tracker is None:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                             (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()
                h, w = frame.shape[:2]
                max_conf = 0
                best_box = None

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    class_id = int(detections[0, 0, i, 1])

                    if confidence > CONFIDENCE_THRESHOLD and CLASS_NAMES[class_id] == "person":
                        box = detections[0, 0, i, 3:7] * [w, h, w, h]
                        (x1, y1, x2, y2) = box.astype("int")
                        if confidence > max_conf:
                            max_conf = confidence
                            best_box = (x1, y1, x2 - x1, y2 - y1)

                if best_box:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, best_box)
                    with bbox_lock:
                        last_bbox = best_box

            elif tracker:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w_, h_ = [int(v) for v in bbox]
                    with bbox_lock:
                        last_bbox = (x, y, w_, h_)
                else:
                    tracker = None

        except Exception as e:
            print("Detection/tracking error:", e)

def generate():
    global output_frame, last_bbox

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            # Overlay current tracking box
            with bbox_lock:
                if last_bbox:
                    x, y, w, h = last_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the current frame for web stream
            with lock:
                output_frame = frame.copy()

            _, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.1)  # 10 FPS

        except Exception as e:
            print("Streaming error:", e)
            break

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Visit /view to see the live video stream."

@app.route('/view')
def view():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drone Person Tracking</title>
        <style>
            body { background-color: #111; color: #eee; font-family: Arial, sans-serif; text-align: center; }
            h1 { margin-top: 20px; }
            img { border: 2px solid #555; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Drone Person Tracking Stream</h1>
        <img src="/video_feed" width="640" height="480" />
    </body>
    </html>
    """

if __name__ == '__main__':
    t = threading.Thread(target=detect_and_track)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=8000, threaded=True)
