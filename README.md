# MediaPipe MQTT
[MediaPipe](https://google.github.io/mediapipe/) examples which stream their detections to a MQTT broker to be used in other applications.

### Install & Run

Currently this is only tested on Windows and MacOS. It's recommended to use Python3 (`>3.7`) and a virtual environment.

```bash
python install -r requirements.txt
```

To run an example use the basic python command to start up the script.

```bash
# start pose detection with webcam 0
python pose.py --input 0

# start pose detection with video
python pose.py --input yoga.mp4
```

Other parameters are documented in the following list or algorithm specific.

- **input** - The video input path or video camera id (default `0`)
- **min-detection-confidence** - Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful. (default `0.5`)
- **min-tracking-confidence** - Minimum confidence value ([0.0, 1.0]) to be considered tracked successfully. (default `0.5`)
- **ip** - MQTT Broker address to send to (default `127.0.0.1`)
- **port** - MQTT port to send to (default `7500`)

### Full-Body Pose Landmark Model (BlazePose Tracker)
The landmark model currently included in MediaPipe Pose predicts the location of 33 full-body landmarks (see figure below), each with (`x, y, z, visibility`). Note that the z value should be discarded as the model is currently not fully trained to predict depth, but this is something we have on the roadmap.

![Pose Description](readme/pose_tracking_full_body_landmarks.png)

*[Reference: mediapipe/solutions/pose](https://google.github.io/mediapipe/solutions/pose#pose-landmark-model-blazepose-tracker)*

**Additional Parameters**

```
--model-complexity MODEL_COMPLEXITY
                      Set model complexity (0=Light, 1=Full, 2=Heavy).
--no-smooth-landmarks
                      Disable landmark smoothing.
--static-image-mode   Enables static image mode.
```


### Running with Tensorflow cuda

Install Tensorflow and cuda with:

    pip install tensorflow[and-cuda]

Verify the GPU setup:

    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


#### Format

- `count` - Indicates how many poses are detected (currently only `0` or `1`)
- list of landmarks (`33` per pose) (if pose has been detected)
    - `x` - X-Position of the landmark
    - `y` - Y-Position of the landmark
    - `z` - Z-Position of the landmark
    - `visibility` - Visibility of the landmark

```
/detections/<id>/pose/landmarks/<landmark>/ [x, y, z, visibility]

#detailed landmarks
/detections/<id>/pose/landmarks
    /nose
        /x
        /y
        /z
        /visibility
    /left_eye
        /x
        /y
        /z
        /visibility
    /right_eye
        /x
        /y
        /z
        /visibility
    /left_ear
        /x
        /y
        /z
        /visibility
    /right_ear
        /x
        /y
        /z
        /visibility
    /left_shoulder
        /x
        /y
        /z
        /visibility
    /right_shoulder
        /x
        /y
        /z
        /visibility
    /left_elbow
        /x
        /y
        /z
        /visibility
    /right_elbow
        /x
        /y
        /z
        /visibility
    /left_wrist
        /x
        /y
        /z
        /visibility
    /right_wrist
        /x
        /y
        /z
        /visibility
    /left_hip
        /x
        /y
        /z
        /visibility
    /right_hip
        /x
        /y
        /z
        /visibility
    /left_knee
        /x
        /y
        /z
        /visibility
    /right_knee
        /x
        /y
        /z
        /visibility

```

### Hand Detection
The [hand detection model](https://google.github.io/mediapipe/solutions/hands.html) is able to detect and track 21 3D landmarks.

#### Format

- `count` - Indicates how many hands are detected
- list of landmarks (`21` per hand) (if hands has been detected)
    - `x` - X-Position of the landmark
    - `y` - Y-Position of the landmark
    - `z` - Z-Position of the landmark
    - `visibility` - Visibility of the landmark

```
/mediapipe/hands [count, x, y, z, visibility, x, y, z, visibility ...]
```

### Face Detection
The [face detection model](https://google.github.io/mediapipe/solutions/face_detection.html) is able to detect multiple faces and 5 keypoints. At the moment only the bounding box is sent over OSC.

#### Format
All values are normalized to the image width and height.

- `count` - Indicates how many faces are detected
- list of one bounding box per face (if faces has been detected)
    - `xmin` - X-Position of the top-left bounding box anchor
    - `ymin` - Y-Position of the top-left bounding box anchor
    - `width` - Width of the bounding box
    - `height` - Height of the bounding box
    - `score` - Confidence score of the bounding box

```
/mediapipe/faces [count, xmin, ymin, width, height, score, xmin, ymin, width, height, score ...]
```

### Face Mesh
tbd

### Examples

Currently, there are very basic receiver examples for processing. Check out the [examples](examples) folder.

### About
* Example code and documentation adapted from [google/mediapipe](https://google.github.io/mediapipe/solutions/)