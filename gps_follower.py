#!/usr/bin/env python3
"""
A Flask server that receives GPS locations via HTTP POST and uses DroneKit
to command an ArduPilot-controlled drone to fly to the specified location at 2 meters altitude.
"""

from flask import Flask, request, jsonify
import threading
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

app = Flask(__name__)

# Connect to the vehicle (adjust the connection string as needed for your setup)
connection_string = '/dev/ttyAMA0'
print(f"Connecting to vehicle on: {connection_string}")
vehicle = connect(connection_string, baud=57600, wait_ready=True)

def arm_and_takeoff(target_altitude):
    """
    Arms the drone and takes off to a specified altitude.
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

    # Wait until the vehicle reaches a safe altitude.
    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        print(f"Current altitude: {current_alt:.2f} m")
        if current_alt >= target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def fly_to_location(lat, lon, alt=5):
    """
    Commands the drone to fly to the specified latitude, longitude, and altitude.
    Here, the altitude is set to 2 meters above the ground (or the specified GPS location).
    """
    # Ensure the drone is in GUIDED mode
    if vehicle.mode.name != "GUIDED":
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)
    
    target_location = LocationGlobalRelative(lat, lon, alt)
    print(f"Commanding flight to: Lat {lat}, Lon {lon}, Alt {alt}")
    vehicle.simple_goto(target_location)

@app.route('/location', methods=['POST'])
def handle_location():
    """
    Receives POST requests with JSON data that must include 'latitude' and 'longitude'.
    Launches a new thread to command the drone to fly to the received location at 2 m altitude.
    """
    if request.is_json:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        if latitude is None or longitude is None:
            return jsonify({"error": "Missing latitude or longitude"}), 400
        
        # Run the flight command in a separate thread to prevent blocking.
        threading.Thread(target=fly_to_location, args=(latitude, longitude)).start()
        return jsonify({"status": "Command received, flying to new location at 2 meters altitude"}), 200
    else:
        return jsonify({"error": "Request must be in JSON format"}), 400

if __name__ == '__main__':
    # Uncomment the following line if you need to arm and take off first.
    # arm_and_takeoff(2)

    # Start the Flask server to listen on all network interfaces at port 5000.
    app.run(host='0.0.0.0', port=5000)
