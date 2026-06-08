"""
main.py - Human Motion Capture pipeline.

Integrates all modules: capture, detection, kinematics, 2D visualization,
Three.js 3D visualization (via WebSocket), and BVH export.

Usage:
    python main.py --source 0              # webcam (default)
    python main.py --source video.mp4      # video file
    python main.py --source image.jpg      # static image
    python main.py --source 0 --record output.bvh   # webcam + BVH recording
    python main.py --source 0 --no-3d      # disable 3D WebSocket server

Controls (video/webcam mode):
    Q - Quit
    R - Toggle BVH recording (start/stop)
"""

import argparse
import sys
import os
import cv2
import time
import webbrowser

from capture import VideoCapture, ImageCapture, is_image
from detector import PoseDetector
from kinematics import compute_all_angles
from visualizer_2d import Visualizer2D
from bvh_exporter import BVHExporter


def _print(msg: str) -> None:
    """Print with flush for Windows compatibility."""
    print(msg, flush=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Human Motion Capture - Realtime skeleton detection & BVH export"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: camera index (0,1,...), video file, or image file (.jpg/.png)"
    )
    parser.add_argument(
        "--record", type=str, default=None,
        help="BVH output file path (auto-start recording)"
    )
    parser.add_argument(
        "--no-3d", action="store_true",
        help="Disable 3D WebSocket server"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="WebSocket server port (default: 8765)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to pose_landmarker .task model file"
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Camera resolution width"
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Camera resolution height"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open browser for 3D viewer"
    )
    return parser.parse_args()


def _start_3d_server(args):
    """Start the 3D WebSocket server and optionally open browser."""
    try:
        from server import PoseServer
        pose_server = PoseServer(port=args.port)
        pose_server.start()
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}")
        return pose_server
    except Exception as e:
        _print(f"[WARN] Could not start 3D server: {e}")
        return None


def run_image_mode(args, source: str):
    """
    Process a single static image: detect skeleton, show overlay, wait for keypress.
    """
    _print(f"[INIT] Image mode: {source}")

    # Use IMAGE running mode (no tracking)
    detector_kwargs = {"running_mode": "IMAGE"}
    if args.model:
        detector_kwargs["model_path"] = args.model

    cap = ImageCapture(source)
    detector = PoseDetector(**detector_kwargs)
    vis2d = Visualizer2D()

    # Start 3D server if enabled
    pose_server = None
    if not args.no_3d:
        pose_server = _start_3d_server(args)

    cap.open()
    _print(f"[INIT] Image size: {cap.frame_size}")

    for frame in cap:
        detection = detector.detect(frame)

        angles = None
        if detection is not None:
            angles = compute_all_angles(
                detection["landmarks_3d"],
                detection["visibility"],
                vis_threshold=0.3,
            )
            _print(f"[INFO] Detected {len(detection['landmarks_2d'])} landmarks")
            if angles:
                _print("[INFO] Joint angles:")
                for name, deg in sorted(angles.items()):
                    _print(f"  {name:20s}: {deg:.1f} deg")
        else:
            _print("[WARN] No person detected in image")

        # Draw 2D overlay
        vis2d.draw(frame, detection, angles)

        # Send to 3D viewer
        if pose_server and detection:
            pose_server.send_pose(detection, angles)
            time.sleep(0.5)  # Give browser time to connect and render
            pose_server.send_pose(detection, angles)  # Resend after connect

        # Show result and wait
        cv2.imshow("Motion Capture - Image", frame)
        _print("[INFO] Press any key to close.")
        cv2.waitKey(0)

    # Cleanup
    cap.close()
    detector.close()
    if pose_server:
        pose_server.stop()
    cv2.destroyAllWindows()
    _print("[EXIT] Done.")


def run_video_mode(args, source):
    """
    Process webcam or video file: real-time detection with all features.
    """
    _print(f"[INIT] Video mode, source: {source}")

    detector_kwargs = {}
    if args.model:
        detector_kwargs["model_path"] = args.model

    cap = VideoCapture(source, width=args.width, height=args.height)
    detector = PoseDetector(**detector_kwargs)
    vis2d = Visualizer2D()
    bvh = BVHExporter(fps=30.0)

    # Start 3D WebSocket server
    pose_server = None
    if not args.no_3d:
        pose_server = _start_3d_server(args)

    # Auto-start recording if --record is specified
    bvh_path = args.record
    if bvh_path:
        bvh.start_recording()
        _print(f"[REC] Recording started -> {bvh_path}")

    # --- Main loop ---
    cap.open()
    _print(f"[INIT] Resolution: {cap.frame_size}, FPS: {cap.fps}")
    _print("[INIT] Controls: Q=Quit, R=Toggle BVH recording")
    _print("=" * 50)

    try:
        for frame in cap:
            # 1. Detect pose
            detection = detector.detect(frame)

            # 2. Compute joint angles
            angles = None
            if detection is not None:
                angles = compute_all_angles(
                    detection["landmarks_3d"],
                    detection["visibility"],
                    vis_threshold=0.3,
                )

            # 3. Record BVH frame
            if bvh.is_recording and detection is not None:
                bvh.add_frame(detection["landmarks_3d"])

            # 4. Draw 2D overlay
            vis2d.draw(frame, detection, angles)

            # Draw recording indicator
            if bvh.is_recording:
                h, w = frame.shape[:2]
                cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"REC [{bvh.frame_count}]",
                    (w - 140, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2,
                )

            cv2.imshow("Motion Capture - 2D", frame)

            # 5. Send to 3D viewer via WebSocket
            if pose_server is not None:
                pose_server.send_pose(detection, angles)

            # 6. Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                _print("\n[EXIT] Quit requested.")
                break
            elif key == ord("r"):
                if bvh.is_recording:
                    bvh.stop_recording()
                    if bvh_path:
                        bvh.save(bvh_path)
                    else:
                        fname = f"mocap_{int(time.time())}.bvh"
                        bvh.save(fname)
                        bvh_path = None
                    _print(f"[REC] Recording stopped. {bvh.frame_count} frames saved.")
                else:
                    if bvh_path is None:
                        bvh_path = f"mocap_{int(time.time())}.bvh"
                    bvh.start_recording()
                    _print(f"[REC] Recording started -> {bvh_path}")

    except KeyboardInterrupt:
        _print("\n[EXIT] Interrupted.")

    finally:
        # Save any unsaved recording
        if bvh.is_recording and bvh.frame_count > 0:
            bvh.stop_recording()
            save_path = bvh_path or f"mocap_{int(time.time())}.bvh"
            bvh.save(save_path)

        # Cleanup
        cap.close()
        detector.close()
        if pose_server is not None:
            pose_server.stop()
        cv2.destroyAllWindows()
        _print("[EXIT] Resources released. Goodbye!")


def main():
    args = parse_args()

    # Determine source type
    source_str = args.source
    try:
        source = int(source_str)
    except ValueError:
        source = source_str

    # Route to appropriate mode
    if isinstance(source, str) and is_image(source):
        run_image_mode(args, source)
    else:
        run_video_mode(args, source)


if __name__ == "__main__":
    main()
