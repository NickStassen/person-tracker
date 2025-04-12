#!/usr/bin/env python3
"""
Drone Control with Ultrasonic Sensors Integration

This script integrates ultrasonic sensors into the drone control
application. Two ultrasonic sensors are used: one facing forward
(for detecting objects ahead) and one facing upward (for detecting
the ceiling). Their readings are acquired continuously in a separate
thread. The script also performs person detection/tracking using a
computer vision model, provides a Flask-based web interface for
video streaming and flight commands, and uses DroneKit for vehicle control.

Requirements:
- RPi.GPIO for ultrasonic sensor communication.
- OpenCV for computer vision tasks.
- DroneKit for interacting with the vehicle.
- Flask for the web server.
"""

import RPi.GPIO as GPIO
import time
import cv2
from flask import Flask, Response, request, jsonify
import threading
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative

# ---------------------- Ultrasonic Sensor Setup ---------------------- #
# Sensor facing forward (for obstacle detection)
TRIG_PIN_FORWARD = 23
ECHO_PIN_FORWARD = 24

# Sensor facing upward (for ceiling detection)
TRIG_PIN_UP = 17
ECHO_PIN_UP = 27

# Setup GPIO for ultrasonic sensors
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN_FORWARD, GPIO.OUT)
GPIO.setup(ECHO_PIN_FORWARD, GPIO.IN)
GPIO.setup(TRIG_PIN_UP, GPIO.OUT)
GPIO.setup(ECHO_PIN_UP, GPIO.IN)

def measure_distance(trig_pin: int, echo_pin: int) -> tuple:
    """
    Measure distance using an ultrasonic sensor.

    Args:
        trig_pin (int): GPIO pin connected to the sensor trigger.
        echo_pin (int): GPIO pin connected to the sensor echo.

    Returns:
        tuple: A tuple containing the measured distance in meters and feet.
    """
    # Trigger a 10µs pulse
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)

    # Wait for the echo to start
    pulse_start = time.time()
    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()

    # Wait for the echo to end
    pulse_end = time.time()
    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance_meters = (pulse_duration * 343) / 2  # Speed of sound = 343 m/s
    distance_feet = distance_meters * 3.28084
    return distance_meters, distance_feet

# Global variables to store sensor readings
front_sensor_distance = (0.0, 0.0)
up_sensor_distance = (0.0, 0.0)

def ultrasonic_loop() -> None:
    """
    Continuously measure distances from both ultrasonic sensors.

    Updates global variables with the current measured values and prints the readings.
    """
    global front_sensor_distance, up_sensor_distance
    while True:
        front_sensor_distance = measure_distance(TRIG_PIN_FORWARD, ECHO_PIN_FORWARD)
        up_sensor_distance = measure_distance(TRIG_PIN_UP, ECHO_PIN_UP)
        print(f"Front Sensor: {front_sensor_distance[0]:.2f} m, {front_sensor_distance[1]:.2f} ft")
        print(f"Up Sensor: {up_sensor_distance[0]:.2f} m, {up_sensor_distance[1]:.2f} ft")
        time.sleep(2)  # Adjust the delay as necessary

# ---------------------- Drone Vision & Control Setup ---------------------- #
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
CONFIG_PATH = "MobileNetSSD_deploy.prototxt"
CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

CONFIDENCE_THRESHOLD = 0.5
DETECTION_INTERVAL = 15
CENTER_TOLERANCE = 30
ADJUSTMENT_STEP = 0.00002

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 2000)
cap.set(cv2.CAP_PROP_CONTRAST, 30)
cap.set(cv2.CAP_PROP_SATURATION, 30)

net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
tracker = None
bbox_lock = threading.Lock()
last_bbox = None
frame_count = 0
MODE = "gps"  # Modes: "gps" or "vision"

# Connect to the vehicle
connection_string = '/dev/ttyAMA0'
print(f"Connecting to vehicle on: {connection_string}")
vehicle = connect(connection_string, baud=57600, wait_ready=True)

def detect_and_track() -> None:
    """
    Detect and track a person using the vision system.

    Every DETECTION_INTERVAL frames the network performs a detection. Between detections,
    a tracker is used to follow the person. If the person is off-center,
    the drone’s location is updated accordingly.
    """
    global tracker, frame_count, last_bbox, output_frame
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            if MODE != "follow":
                continue

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
                    # Center tracking adjustment
                    cx = x + w_ // 2
                    cy = y + h_ // 2
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2
                    dx = cx - frame_center_x
                    dy = cy - frame_center_y

                    if abs(dx) > CENTER_TOLERANCE or abs(dy) > CENTER_TOLERANCE:
                        current_location = vehicle.location.global_relative_frame
                        new_lat = current_location.lat
                        new_lon = current_location.lon

                        if abs(dx) > CENTER_TOLERANCE:
                            new_lon += ADJUSTMENT_STEP if dx > 0 else -ADJUSTMENT_STEP
                        if abs(dy) > CENTER_TOLERANCE:
                            new_lat -= ADJUSTMENT_STEP if dy > 0 else -ADJUSTMENT_STEP

                        target = LocationGlobalRelative(new_lat, new_lon, current_location.alt)
                        vehicle.simple_goto(target)
                else:
                    tracker = None
        except Exception as e:
            print("Detection/tracking error:", e)

def generate() -> bytes:
    """
    Generate a video stream by encoding frames as JPEG images.

    Returns:
        A byte stream yielding the video frames.
    """
    global output_frame, last_bbox
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            with bbox_lock:
                if MODE == "follow" and last_bbox:
                    x, y, w, h = last_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            with lock:
                output_frame = frame.copy()

            _, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
        except Exception as e:
            print("Streaming error:", e)
            break

def fly_to_location(lat: float, lon: float, alt: float = 5) -> None:
    """
    Command the vehicle to fly to a specified location.

    Args:
        lat (float): Target latitude.
        lon (float): Target longitude.
        alt (float, optional): Target altitude. Defaults to 5.
    """
    if vehicle.mode.name != "GUIDED":
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)

    target_location = LocationGlobalRelative(lat, lon, alt)
    print(f"Commanding flight to: Lat {lat}, Lon {lon}, Alt {alt}")
    vehicle.simple_goto(target_location)

def arm_and_takeoff(target_altitude: float) -> None:
    """
    Arm the vehicle and take off to a target altitude.

    Args:
        target_altitude (float): The target altitude to reach.
    """
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

# ---------------------- Flask Endpoints ---------------------- #
@app.route('/location', methods=['POST'])
def handle_location():
    global MODE
    if request.is_json:
        if MODE != "gps":
            return jsonify({"error": "Location command only works in GPS mode"}), 400

        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        if latitude is None or longitude is None:
            return jsonify({"error": "Missing latitude or longitude"}), 400

        threading.Thread(target=fly_to_location, args=(latitude, longitude)).start()
        return jsonify({"status": "Command received, flying to new location at 2 meters altitude"}), 200
    else:
        return jsonify({"error": "Request must be in JSON format"}), 400

@app.route('/command', methods=['POST'])
def switch_mode():
    global MODE
    if request.is_json:
        print(f"Received command: {request.data}")
        data = request.get_json()
        mode = data.get("command")
        if mode in ["follow", "standby"]:
            MODE = mode
            if mode == "standby":
                # Switch to LOITER so the drone holds its position
                vehicle.mode = VehicleMode("LOITER")
            else:  # follow mode
                vehicle.mode = VehicleMode("GUIDED")
            return jsonify({"status": f"Mode switched to {mode}"}), 200
        return jsonify({"error": "Invalid mode"}), 400
    return jsonify({"error": "Request must be JSON"}), 400


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

# ---------------------- Main Execution ---------------------- #
if __name__ == '__main__':
    try:
        # Start ultrasonic sensor polling thread
        ultrasonic_thread = threading.Thread(target=ultrasonic_loop, daemon=True)
        ultrasonic_thread.start()

        # Start the person detection and tracking thread
        tracking_thread = threading.Thread(target=detect_and_track, daemon=True)
        tracking_thread.start()

        # Run the Flask web server
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        GPIO.cleanup()
