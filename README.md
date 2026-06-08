# Human Motion Capture

Human Motion Capture demo using Python, MediaPipe Pose, OpenCV, FastAPI, WebSocket, and Three.js. The backend extracts 33 human pose landmarks from webcam or video, classifies gestures, computes FK/IK kinematics, logs results to CSV, and streams a realtime 3D skeleton to the browser.

![Demo preview](docs/demo-screenshot.svg)

## Features

- Webcam/video capture with MediaPipe Pose Landmarker.
- 33 landmarks with 2D pixel coordinates, 3D world coordinates, and joint visibility confidence.
- OpenCV overlay with skeleton, FPS, current pose, joint angles, and per-joint confidence values.
- Rule-based gesture recognition for `STANDING`, `SITTING`, `LEFT_HAND_UP`, `RIGHT_HAND_UP`, `BOTH_HANDS_UP`, and `BENDING_FORWARD`.
- CSV recognition log at `logs/gesture_log.csv` with `timestamp`, `detected_pose`, and `confidence`.
- Forward kinematics summary: joint angles, bone lengths, and segment directions for arms, legs, and spine.
- Inverse kinematics with FABRIK for left/right arm chains.
- FastAPI app with WebSocket endpoint `/ws/pose`.
- Three.js frontend with `LineSegments` bones, `SphereGeometry` joints, OrbitControls, and a simple low-poly `SkinnedMesh + Skeleton` humanoid.
- UI controls for pause/resume, webcam source, video upload, 2D overlay, 3D skeleton, humanoid mesh, IK overlay, grid, and hip-centering.
- WebSocket optimization: frames are broadcast only when landmark movement exceeds a delta threshold, the gesture changes, detection state changes, or a heartbeat is due.

## Project Structure

```text
main.py                    FastAPI app, capture runtime, /ws/pose, /video_feed
pose_extractor.py          MediaPipe Pose wrapper returning Joint/PoseFrame
gesture_classifier.py      Rule-based gesture/posture classifier
kinematics.py              Joint angles, FK summary, FABRIK IK
utils.py                   Shared math, landmark, JPEG, and delta helpers
visualizer_2d.py           OpenCV skeleton overlay
motion_filter.py           EMA landmark smoothing
static/index.html          Browser UI shell
static/skeleton_renderer.js Three.js scene, joints, bones, IK overlay
static/humanoid_mesh.js    Low-poly SkinnedMesh humanoid
static/websocket_client.js WebSocket reconnect/FPS client
static/controls.js         Browser UI controls
tests/                     Unit tests for kinematics, IK, gesture rules
```

## Install

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_env.py
```

The MediaPipe model file `pose_landmarker.task` must stay in the project root.

## Run Demo

Start the backend:

```powershell
uvicorn main:app --reload
```

Open:

```text
http://localhost:8000
```

Expected result: the browser shows a realtime 3D skeleton driven by webcam pose data. The 2D OpenCV overlay appears as an MJPEG panel in the browser and displays FPS, current pose, joint angles, and joint confidence scores.

If no person is detected, both backend and frontend display `No person detected` and keep the stream alive.

## Frontend Controls

- Webcam button: use the camera index from the number input.
- Upload button: upload a local video and switch the backend source to it.
- Pause button: pause/resume capture processing.
- 2D overlay: toggle the OpenCV MJPEG panel.
- 3D skeleton: toggle line-and-sphere skeleton.
- Humanoid: toggle the low-poly `SkinnedMesh` body.
- IK: toggle the FABRIK arm overlay.
- Grid: toggle the ground grid.
- Crosshair: toggle hip-centered normalization.
- Reset: reset OrbitControls camera.

## Gesture Rules

The classifier is intentionally rule-based so it is easy to explain in the report:

| Gesture | Rule summary |
| --- | --- |
| `STANDING` | Knee and hip angles are mostly extended. |
| `SITTING` | Knee bend and/or hip bend is high. |
| `LEFT_HAND_UP` | Left wrist is above left shoulder in image coordinates. |
| `RIGHT_HAND_UP` | Right wrist is above right shoulder in image coordinates. |
| `BOTH_HANDS_UP` | Both wrist-up rules are true. |
| `BENDING_FORWARD` | Torso leans away from vertical or nose drops toward shoulders. |

Each score is weighted by MediaPipe landmark visibility so occluded joints reduce confidence.

## Kinematics

Forward kinematics is computed from MediaPipe joint positions:

- Joint angles: shoulder, elbow, hip, and knee are calculated with the angle between three 3D landmarks.
- Bone lengths: each skeleton segment length is measured in MediaPipe world coordinates.
- Segment directions: each parent-to-child vector is normalized so child positions can be reconstructed from root position, direction, and bone length.

Inverse kinematics uses FABRIK for each arm chain:

1. Keep the shoulder root fixed.
2. Move the wrist end effector to the target.
3. Run a backward pass from wrist to shoulder while preserving bone lengths.
4. Run a forward pass from shoulder to wrist while preserving bone lengths.
5. Repeat until the wrist reaches the target tolerance or max iterations is reached.

The browser can toggle the IK overlay to compare MediaPipe arm landmarks against the FABRIK-solved target chain.

## Data Log

Gesture recognition is written to:

```text
logs/gesture_log.csv
```

Columns:

```text
timestamp,detected_pose,confidence
```

## Test

```powershell
python -m unittest discover -s tests
```

Current tests cover angle calculation, FK bone lengths, two-bone IK compatibility, FABRIK length preservation/target reach, EMA smoothing, and gesture classification.

## Webcam Smoke Test

To test only MediaPipe extraction and OpenCV overlay:

```powershell
python pose_extractor.py
```

Press `q` to close the window.
