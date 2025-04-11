import cv2
from flask import Flask, Response
import threading

"""
Headless real-time person detection and tracking using MobileNet-SSD + KCF,
served as an MJPEG stream on port 8000.
"""

MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
CONFIG_PATH = "MobileNetSSD_deploy.prototxt"
CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

CONFIDENCE_THRESHOLD = 0.5
DETECTION_INTERVAL = 15  # Detect every N frames

net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

tracker = None
frame_count = 0

def detect_and_track():
    global output_frame, lock, tracker, frame_count

    while cap.isOpened():
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
                cv2.rectangle(frame, (best_box[0], best_box[1]),
                              (best_box[0] + best_box[2], best_box[1] + best_box[3]),
                              (0, 255, 0), 2)

        elif tracker:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w_, h_ = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w_, y + h_), (255, 0, 0), 2)
            else:
                tracker = None

        with lock:
            output_frame = frame.copy()

def generate():
    global output_frame, lock

    while True:
        with lock:
            if output_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Go to /video_feed to view the stream."

if __name__ == '__main__':
    t = threading.Thread(target=detect_and_track)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=8000, threaded=True)
