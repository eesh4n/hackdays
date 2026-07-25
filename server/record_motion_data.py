"""
MOTION DATA RECORDER
=====================

Records normalized pose landmark positions to a CSV file so we can look
at what a jump and a duck actually look like in the numbers, then pick
detection thresholds for the subway-surfer-style jump/duck detector.

Move around normally for a few seconds first (baseline), then do a few
clean jumps and a few clean ducks, then press 'q' to stop.

Writes: server/motion_log.csv
"""

import csv
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "pose_landmarker_lite.task")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "motion_log.csv")

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def mid_y(landmarks, a, b):
    return (landmarks[a].y + landmarks[b].y) / 2


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

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

    window_name = "Motion Recorder"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time_ms = int(time.time() * 1000)
    last_timestamp_ms = -1
    frame_num = 0

    csv_file = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "frame", "timestamp_ms",
        "nose_y", "shoulder_y", "hip_y", "knee_y", "ankle_y",
        "torso_len",
    ])

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

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]

                nose_y = lm[NOSE].y
                shoulder_y = mid_y(lm, LEFT_SHOULDER, RIGHT_SHOULDER)
                hip_y = mid_y(lm, LEFT_HIP, RIGHT_HIP)
                knee_y = mid_y(lm, LEFT_KNEE, RIGHT_KNEE)
                ankle_y = mid_y(lm, LEFT_ANKLE, RIGHT_ANKLE)
                torso_len = hip_y - shoulder_y

                writer.writerow([
                    frame_num, timestamp_ms,
                    f"{nose_y:.4f}", f"{shoulder_y:.4f}", f"{hip_y:.4f}",
                    f"{knee_y:.4f}", f"{ankle_y:.4f}", f"{torso_len:.4f}",
                ])

                lines = [
                    f"nose_y:     {nose_y:.3f}",
                    f"shoulder_y: {shoulder_y:.3f}",
                    f"hip_y:      {hip_y:.3f}",
                    f"knee_y:     {knee_y:.3f}",
                    f"ankle_y:    {ankle_y:.3f}",
                    f"torso_len:  {torso_len:.3f}",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(
                        frame, text, (20, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2, cv2.LINE_AA
                    )

                for idx in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
                            LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
                    x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            else:
                cv2.putText(
                    frame, "No person detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )

            cv2.putText(
                frame, f"Recording... frame {frame_num}  (press q to stop)",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)
            frame_num += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print(f"Saved {frame_num} frames to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
