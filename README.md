# Human Motion Capture

Project for recognizing human gestures and motion from webcam, video, or image input. The pipeline uses MediaPipe Pose Landmarker to extract a 33-point skeleton, computes kinematics data, streams motion to a WebGL/Three.js viewer, and can export the captured motion to BVH, JSON, or CSV.

## Features

- Detect human pose from webcam, video files, or static images.
- Draw 2D skeleton overlay with OpenCV.
- Compute major joint angles: shoulders, elbows, hips, and knees.
- Compute forward-kinematics data: skeleton chains, bone lengths, and segment directions.
- Demonstrate inverse kinematics with two-bone arm/leg chains.
- Smooth 3D landmarks with an EMA filter to reduce jitter.
- Stream pose data to a local FastAPI WebSocket server.
- Render a 3D skeleton in a WebGL/Three.js viewer.
- Export animation to BVH for Blender or other 3D tools.
- Export raw landmarks and angles to JSON or CSV for reports and analysis.

## Requirements

- Python 3.10 or newer.
- Webcam for realtime capture, or a video/image file.
- `pose_landmarker.task` in the project root.

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Check the environment:

```powershell
python check_env.py
```

## Run

Webcam realtime capture:

```powershell
python main.py --source 0
```

Video file:

```powershell
python main.py --source path\to\video.mp4
```

Static image:

```powershell
python main.py --source path\to\image.jpg
```

Disable the 3D viewer:

```powershell
python main.py --source 0 --no-3d
```

Use another server port:

```powershell
python main.py --source 0 --port 9000
```

## Controls

OpenCV window:

- `Q`: quit.
- `R`: start or stop BVH recording.

WebGL viewer:

- `#`: toggle grid.
- `A`: toggle joint angle panel.
- `C`: toggle hip-centered skeleton normalization.
- `IK`: toggle inverse-kinematics demo overlay.
- `R`: reset camera.

## Export

Auto-record BVH from startup:

```powershell
python main.py --source 0 --record output.bvh
```

Export raw data:

```powershell
python main.py --source 0 --export-json output\motion.json --export-csv output\motion.csv
```

Smoothing is enabled by default:

```powershell
python main.py --source 0 --smooth-alpha 0.45
```

Disable smoothing:

```powershell
python main.py --source 0 --no-smoothing
```

`smooth-alpha` is the EMA coefficient. Lower values are smoother but add more delay. `1.0` behaves almost like no smoothing.

## Project Structure

- `main.py`: main pipeline and CLI.
- `detector.py`: MediaPipe Pose Landmarker wrapper.
- `capture.py`: webcam, video, and image capture helpers.
- `visualizer_2d.py`: OpenCV skeleton overlay.
- `kinematics.py`: joint angles, FK summary, bone lengths, and IK demo solver.
- `motion_filter.py`: temporal landmark smoothing.
- `bvh_exporter.py`: BVH animation export.
- `data_exporter.py`: JSON/CSV data export.
- `server.py`: FastAPI server and WebSocket stream.
- `static/index.html`: Three.js WebGL viewer.
- `check_env.py`: dependency and model checker.
- `tests/`: focused tests for kinematics and smoothing.

## Suggested Demo Flow

1. Run `python check_env.py`.
2. Run `python main.py --source 0`.
3. Show the OpenCV 2D skeleton and joint-angle overlay.
4. Show the browser WebGL skeleton moving in realtime.
5. Toggle `IK` in the viewer to show the inverse-kinematics demo overlay.
6. Press `R` in the OpenCV window to record BVH, then stop with `R` again.
7. Run with `--export-json` or `--export-csv` and show that each frame has landmarks and angles.

## Tests

Run:

```powershell
python -m unittest discover -s tests
```

The tests cover angle calculation, bone-length extraction, two-bone IK length preservation, and EMA smoothing.
