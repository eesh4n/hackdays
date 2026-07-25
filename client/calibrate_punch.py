"""
Punch calibration tool -- NOT used by the real game. Records the right
wrist's actual disp/speed/straightness numbers (the same metrics
player_b_tracker.py's punch detector uses) while you throw real punches,
so PUNCH_MIN_NET_DISPLACEMENT / PUNCH_MIN_PEAK_SPEED / PUNCH_MIN_STRAIGHTNESS
can be set from real data instead of guesses.

Run: python calibrate_punch.py
Throw ~6-8 real punches at whatever pace feels natural over the session,
with a brief pause between each so they show up as separate peaks. Press
'q' in the window to stop early.
"""
import time

import cv2

from player_b_tracker import PlayerBTracker

DURATION_SEC = 15
PEAK_MIN_SPEED = 0.05  # ignore near-zero noise when looking for peaks
PEAK_MERGE_WINDOW_SEC = 0.2  # peaks closer together than this = same punch


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("FAIL: could not open camera")
        return

    tracker = PlayerBTracker()
    print(f"Throw ~6-8 real punches over the next {DURATION_SEC}s, pausing briefly between each.")
    print("Press 'q' in the window to stop early.")

    samples = []  # (t, disp, speed, straightness)
    start = time.time()

    while time.time() - start < DURATION_SEC:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker.process_frame(rgb)
        samples.append((
            time.time() - start,
            tracker._last_net_displacement,
            tracker._last_peak_speed,
            tracker._last_straightness,
        ))

        tracker.debug_overlay(frame)
        remaining = DURATION_SEC - (time.time() - start)
        cv2.putText(frame, f"CALIBRATING -- {remaining:.1f}s left, throw punches now",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Punch calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    # Local peaks in speed above a noise floor = candidate punches.
    peaks = []
    for i in range(1, len(samples) - 1):
        t, disp, speed, straight = samples[i]
        if speed < PEAK_MIN_SPEED:
            continue
        if speed >= samples[i - 1][2] and speed >= samples[i + 1][2]:
            peaks.append((t, disp, speed, straight))

    # Merge peaks within the same short window into one (multiple local
    # maxima from a single punch), keeping the highest-speed sample.
    merged = []
    for p in peaks:
        if merged and p[0] - merged[-1][0] < PEAK_MERGE_WINDOW_SEC:
            if p[2] > merged[-1][2]:
                merged[-1] = p
        else:
            merged.append(p)

    print("\n--- DETECTED PUNCH-LIKE PEAKS ---")
    if not merged:
        print("No peaks detected above the noise floor -- try throwing punches")
        print("more clearly, or closer to the camera, and rerun.")
        return

    for i, (t, disp, speed, straight) in enumerate(merged):
        print(f"  punch #{i + 1} at t={t:.2f}s: disp={disp:.3f}  speed={speed:.2f}  straightness={straight:.2f}")

    disps = [p[1] for p in merged]
    speeds = [p[2] for p in merged]
    straights = [p[3] for p in merged]

    print("\n--- OBSERVED STATS ---")
    print(f"disp:        min={min(disps):.3f}  avg={sum(disps) / len(disps):.3f}  max={max(disps):.3f}")
    print(f"speed:       min={min(speeds):.2f}  avg={sum(speeds) / len(speeds):.2f}  max={max(speeds):.2f}")
    print(f"straightness: min={min(straights):.2f}  avg={sum(straights) / len(straights):.2f}  max={max(straights):.2f}")

    # Suggest thresholds a bit below the weakest real punch observed, so
    # your softest legitimate punch still clears the bar with margin.
    suggested_disp = min(disps) * 0.75
    suggested_speed = min(speeds) * 0.75
    suggested_straight = min(straights) * 0.85

    print("\n--- SUGGESTED THRESHOLDS (set these in player_b_tracker.py) ---")
    print(f"PUNCH_MIN_NET_DISPLACEMENT = {suggested_disp:.3f}")
    print(f"PUNCH_MIN_PEAK_SPEED = {suggested_speed:.2f}")
    print(f"PUNCH_MIN_STRAIGHTNESS = {suggested_straight:.2f}")


if __name__ == "__main__":
    main()
