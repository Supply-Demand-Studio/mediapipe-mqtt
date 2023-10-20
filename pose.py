import argparse

import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
import numpy as np
from mediapipe.framework.formats import landmark_pb2


from utils import add_default_args, get_video_input

# MQTT Broker Configuration
MQTT_BROKER_HOST = "10.0.0.71"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_PREFIX = "/detections/1/pose/landmarks/"

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


def draw_pose_rect(image, rect, color=(255, 0, 255), thickness=2):
    image_width = image.shape[1]
    image_height = image.shape[0]

    world_rect = [(rect.x_center * image_width, rect.y_center * image_height),
                  (rect.width * image_width, rect.height * image_height),
                  rect.rotation]

    box = cv2.boxPoints(world_rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color, thickness)


def send_pose(client: mqtt.Client,
              landmark_list: landmark_pb2.NormalizedLandmarkList):
    if landmark_list is None:
        return

    for idx, landmark in enumerate(landmark_list.landmark):
        landmark_data = {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z if hasattr(landmark, 'z') else 0.0,
            "visibility": landmark.visibility
        }
        topic = MQTT_TOPIC_PREFIX + str(idx)
        payload = f'{{"x": {landmark_data["x"]}, "y": {landmark_data["y"]}, "z": {landmark_data["z"]}, "visibility": {landmark_data["visibility"]}}}'
        client.publish(topic, payload)

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
        if cv2.waitKey(5) & 0xFF == 27: #ESC
            break
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
