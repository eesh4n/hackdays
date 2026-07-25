"""
BASIC MEDIAPIPE POSE WINDOW
============================

Opens a fullscreen webcam window, detects body pose, and draws each
landmark as a dot with its (x, y) pixel position labeled next to it.
The frame is split into 3 equal vertical lanes and the lane the
person is currently standing in is highlighted and reported.

SETUP (one-time)
-----------------
1. pip install mediapipe opencv-contrib-python

2. Download the pose model file and put it in the SAME folder as
   this script, named exactly: pose_landmarker_lite.task

   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

RUN
---
    python basic_pose_window.py

Press 'q' to quit.
"""

import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from jump_duck_detector import JumpDuckDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "pose_landmarker_lite.task")

LANE_COUNT = 3
LANE_NAMES = ["LEFT", "CENTER", "RIGHT"]

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

# How long a JUMP/DUCK banner stays on screen after it fires.
EVENT_BANNER_MS = 600

# A subset of BlazePose's 33 connections, enough to see the skeleton clearly.
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),   # torso
    (11, 13), (13, 15),                        # left arm
    (12, 14), (14, 16),                        # right arm
    (23, 25), (25, 27), (27, 29), (27, 31),    # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),    # right leg
]


def draw_lanes(frame, lane_width, lane_count, current_lane):
    h, w = frame.shape[:2]

    # Highlight the active lane.
    if current_lane is not None:
        overlay = frame.copy()
        x0 = current_lane * lane_width
        x1 = w if current_lane == lane_count - 1 else x0 + lane_width
        cv2.rectangle(overlay, (x0, 0), (x1, h), (0, 255, 136), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Vertical divider lines between lanes.
    for i in range(1, lane_count):
        x = i * lane_width
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 2)

    # Lane labels + status banner.
    for i in range(lane_count):
        x0 = i * lane_width
        label = LANE_NAMES[i] if i < len(LANE_NAMES) else f"LANE {i + 1}"
        color = (0, 255, 136) if i == current_lane else (200, 200, 200)
        cv2.putText(
            frame, label, (x0 + 20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA
        )

    status = f"Lane: {LANE_NAMES[current_lane] if current_lane is not None else '-'}"
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 136), 2, cv2.LINE_AA)


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Download it from:\n"
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task\n"
            f"and save it in: {SCRIPT_DIR}"
        )

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "MediaPipe Pose"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time_ms = int(time.time() * 1000)
    last_timestamp_ms = -1

    detector = JumpDuckDetector()
    last_event = None
    last_event_time_ms = -EVENT_BANNER_MS

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000) - start_time_ms
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            lane_width = w // LANE_COUNT

            current_lane = None

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # first detected person
                points = [(int(p.x * w), int(p.y * h)) for p in landmarks]

                # Skeleton lines
                for a, b in POSE_CONNECTIONS:
                    cv2.line(frame, points[a], points[b], (0, 255, 136), 2)

                # Dots + position labels for every landmark
                for i, (x, y) in enumerate(points):
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    cv2.putText(
                        frame, f"{i}:({x},{y})", (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
                    )

                # Person's horizontal position = midpoint of the two hips.
                left_hip_x, right_hip_x = points[LEFT_HIP][0], points[RIGHT_HIP][0]
                center_x = (left_hip_x + right_hip_x) // 2
                current_lane = min(center_x // lane_width, LANE_COUNT - 1)

                # Jump/duck detection from normalized (0-1) landmark y-values.
                hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
                shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
                torso_len = hip_y - shoulder_y

                event = detector.update(hip_y, torso_len)
                if event:
                    last_event = event
                    last_event_time_ms = timestamp_ms
                    print(f"[{timestamp_ms}ms] {event}")
            else:
                cv2.putText(
                    frame, "No person detected", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )

            draw_lanes(frame, lane_width, LANE_COUNT, current_lane)

            if last_event and timestamp_ms - last_event_time_ms < EVENT_BANNER_MS:
                color = (0, 200, 255) if last_event == "JUMP" else (255, 100, 0)
                text_size = cv2.getTextSize(last_event, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(
                    frame, last_event, (text_x, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 6, cv2.LINE_AA
                )

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()