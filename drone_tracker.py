import os
import time
import threading
from flask import Flask, request, jsonify, Response
from dronekit import connect, VehicleMode, LocationGlobalRelative
import cv2

# ---------------------------- DroneKit Setup ----------------------------

connection_string = '/dev/ttyAMA0'
print(f"Connecting to vehicle on: {connection_string}")
vehicle = connect(connection_string, baud=57600, wait_ready=True)

# ---------------------------- Flask App Setup ----------------------------

app = Flask(__name__)

# ---------------------------- Global States ----------------------------

MODE = "gps"  # "gps" or "vision"
lock = threading.Lock()

# ---------------------------- Arming and Takeoff ----------------------------

def arm_and_takeoff(target_altitude):
    print("Waiting for vehicle to become armable...")
    while not vehicle.is_armable:
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    print("Arming vehicle...")
    while not vehicle.armed:
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)

    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        print(f"Current altitude: {current_alt:.2f} m")
        if current_alt >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

# ---------------------------- GPS Navigation ----------------------------

def fly_to_location(lat, lon, alt=5):
    if vehicle.mode.name != "GUIDED":
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)

    target_location = LocationGlobalRelative(lat, lon, alt)
    print(f"Commanding flight to: Lat {lat}, Lon {lon}, Alt {alt}")
    vehicle.simple_goto(target_location)

@app.route('/location', methods=['POST'])
def handle_location():
    if request.is_json:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        if latitude is None or longitude is None:
            return jsonify({"error": "Missing latitude or longitude"}), 400

        if MODE == "gps":
            threading.Thread(target=fly_to_location, args=(latitude, longitude)).start()
            return jsonify({"status": "Flying to GPS location"}), 200
        else:
            return jsonify({"status": "Ignored due to vision mode"}), 200
    return jsonify({"error": "Request must be JSON"}), 400

@app.route('/mode', methods=['POST'])
def switch_mode():
    global MODE
    if request.is_json:
        data = request.get_json()
        mode = data.get('mode')
        if mode in ["gps", "vision"]:
            MODE = mode
            return jsonify({"status": f"Mode switched to {MODE}"}), 200
    return jsonify({"error": "Invalid mode or bad request"}), 400

# ---------------------------- Vision Tracking ----------------------------

MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
CONFIG_PATH = "MobileNetSSD_deploy.prototxt"
CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

CONFIDENCE_THRESHOLD = 0.5
DETECTION_INTERVAL = 15
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 2000)
cap.set(cv2.CAP_PROP_CONTRAST, 30)
cap.set(cv2.CAP_PROP_SATURATION, 30)

net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)
tracker = None
frame_count = 0
bbox_lock = threading.Lock()
last_bbox = None
output_frame = None


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
                    with bbox_lock:
                        last_bbox = tuple(map(int, bbox))
                else:
                    tracker = None

            if MODE == "vision" and last_bbox:
                cx = last_bbox[0] + last_bbox[2] // 2
                cy = last_bbox[1] + last_bbox[3] // 2
                offset_x = cx - frame.shape[1] // 2
                offset_y = cy - frame.shape[0] // 2
                print(f"[Vision] Offset: X={offset_x}, Y={offset_y}")
                # Example use: You can add your drone movement logic here

            _, buffer = cv2.imencode('.jpg', frame)
            with lock:
                global output_frame
                output_frame = buffer.tobytes()

        except Exception as e:
            print("Detection/tracking error:", e)


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if output_frame:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
            time.sleep(0.1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Drone Control + Video Feed</h1><p>Use /view or POST to /location or /mode</p>"

@app.route('/view')
def view():
    return '''
    <!DOCTYPE html>
    <html><head><title>Drone Tracking</title></head>
    <body><h1>Live Stream</h1><img src="/video_feed" width="640" height="480" /></body>
    </html>
    '''

if __name__ == '__main__':
    threading.Thread(target=detect_and_track, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
