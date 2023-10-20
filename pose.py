import argparse

import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from utils import add_default_args, get_video_input
import time 
import tensorflow as tf

#Tendsorflow setup fopr GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Configurable parameters for rate limiting
detections_per_second = 10  # Adjust this to the desired rate (e.g., 2 detections per second)
time_interval = 1.0 / detections_per_second  # Calculate the time interval


# MQTT Broker Configuration
MQTT_BROKER_HOST = "10.0.0.71"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_PREFIX = "/detections/1/pose/landmarks/"
DETECTION_ID = 1  # Replace with your desired detection ID
VISIBILITY_THRESHOLD = 0.5

# Initialize MQTT client
client = mqtt.Client()

# Callback function to be called when the client connects to the MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print(f"Connection failed with error code {rc}")

# Callback function to be called when the client disconnects from the MQTT broker
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"Unexpected disconnection. Error code: {rc}")
    else:
        print("Disconnected from MQTT broker")

# Set up the callback functions
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Start the MQTT client loop (if needed)
client.loop_start()

# Connect client to broker
client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# List of landmark names
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def draw_pose_rect(image, rect, color=(255, 0, 255), thickness=2):
    image_width = image.shape[1]
    image_height = image.shape[0]

    world_rect = [(rect.x_center * image_width, rect.y_center * image_height),
                  (rect.width * image_width, rect.height * image_height),
                  rect.rotation]

    box = cv2.boxPoints(world_rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color, thickness)

# Normalized Landmark represents a point in 3D space with x, y, z coordinates. x and y are normalized to [0.0, 1.0] 
def send_pose(client: mqtt.Client,
              landmark_list: landmark_pb2.NormalizedLandmarkList):
    if landmark_list is None:
        return

    #print(landmark_list)
    num_visible_landmarks = 0  # Initialize a counter for the number of visible landmarks

    for idx, landmark in enumerate(landmark_list.landmark):
        if landmark.visibility > VISIBILITY_THRESHOLD:  # Check landmark visibility
            num_visible_landmarks += 1  # Increment the counter for each detected landmark
            landmark_data = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z if hasattr(landmark, 'z') else 0.0,
                "visibility": landmark.visibility
            }
            landmark_name = LANDMARK_NAMES[idx] # Get landmark name
            topic = f"/detections/{DETECTION_ID}/pose/landmarks/{landmark_name}"
            payload = f'{{"x": {landmark_data["x"]}, "y": {landmark_data["y"]}, "z": {landmark_data["z"]}, "visibility": {landmark_data["visibility"]}}}'
            client.publish(topic, payload)
    print(f"Number of detected landmarks: {num_visible_landmarks}")

def main():
    # read arguments
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    parser.add_argument("--model-complexity", default=1, type=int,
                        help="Set model complexity (0=Light, 1=Full, 2=Heavy).")
    parser.add_argument("--no-smooth-landmarks", action="store_false", help="Disable landmark smoothing.")
    parser.add_argument("--static-image-mode", action="store_true", help="Enables static image mode.")
    args = parser.parse_args()

    # setup camera loop
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        smooth_landmarks=args.no_smooth_landmarks,
        static_image_mode=args.static_image_mode,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence)
    cap = cv2.VideoCapture(get_video_input(args.input))

    # fix bug which occurs because draw landmarks is not adapted to upper pose
    connections = mp_pose.POSE_CONNECTIONS

    while cap.isOpened():
        start_time = time.time()  # Record the start time of each iteration
        success, frame = cap.read()
        if not success:
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        #image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # send the pose over osc
        send_pose(client, results.pose_landmarks)

        if hasattr(results, "pose_rect_from_landmarks"):
            draw_pose_rect(image, results.pose_rect_from_landmarks)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, connections)
        # Display the frame
        cv2.imshow("Pose Landmarks", frame)

        end_time = time.time()  # Record the end time of each iteration
        elapsed_time = end_time - start_time  # Calculate the time elapsed during processing

        if elapsed_time < time_interval:
            # If processing took less time than the desired interval, sleep for the difference
            time.sleep(time_interval - elapsed_time)

        if cv2.waitKey(5) & 0xFF == 27: #ESC
            break
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
