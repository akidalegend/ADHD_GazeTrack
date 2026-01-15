import time
import csv
import argparse
import math
from pathlib import Path
from collections import deque

import cv2  # type: ignore
import numpy as np

from gaze_tracking import GazeTrackingMediaPipe as GazeTracking
from analysis import load_calibration_model

def safe_angle(hp, key):
    try:
        return float(hp['angles'].get(key))
    except Exception:
        return float('nan')

def safe_pupil(coords, idx):
    try:
        return float(coords[idx])
    except Exception:
        return float('nan')

def _create_fixation_dot(dot_size, dot_radius, dot_color):
    canvas = np.zeros((dot_size, dot_size, 3), dtype=np.uint8)
    center = (dot_size // 2, dot_size // 2)
    cv2.circle(canvas, center, dot_radius, dot_color, -1)
    return canvas


def main(
    output,
    duration,
    show_dot=False,
    dot_size=700,
    dot_radius=18,
    dot_color=(0, 0, 255),
    calib_model=None,
    smooth_window: int = 8,
    label: str | None = None,
):
    gaze = GazeTracking()
    cap = cv2.VideoCapture(0)
    start = time.time()
    end_time = start + duration if duration > 0 else None

    # If no model provided but label is, try to load automatically
    if calib_model is None and label:
        try:
            calib_model = load_calibration_model(label)
            if calib_model:
                print(f"Loaded calibration model for {label}.")
        except Exception as _e:
            calib_model = None
            print("Proceeding without calibration model.")

    # Smoothing buffers for calibrated gaze
    hist_x: deque = deque(maxlen=int(smooth_window) if smooth_window else 1)
    hist_y: deque = deque(maxlen=int(smooth_window) if smooth_window else 1)

    dot_image = None
    if show_dot:
        dot_size = int(dot_size)
        dot_radius = int(dot_radius)
        dot_image = _create_fixation_dot(dot_size, dot_radius, dot_color)
        cv2.namedWindow('Fixation Dot', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fixation Dot', dot_size, dot_size)
        cv2.imshow('Fixation Dot', dot_image)

    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            't', 'yaw', 'pitch', 'roll',
            'g_horizontal', 'g_vertical', 'is_blinking',
            'left_px', 'left_py', 'right_px', 'right_py',
            'est_gaze_x', 'est_gaze_y', 'smooth_gaze_x', 'smooth_gaze_y'
        ])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = time.time()
            gaze.refresh(frame)
            hp = gaze.head_pose or {}
            yaw = safe_angle(hp, 'yaw')
            pitch = safe_angle(hp, 'pitch')
            roll = safe_angle(hp, 'roll')

            g_h = gaze.horizontal_ratio() if gaze.pupils_located else float('nan')
            g_v = gaze.vertical_ratio() if gaze.pupils_located else float('nan')
            blink = bool(gaze.is_blinking()) if gaze.pupils_located else False

            lp = gaze.pupil_left_coords() or (float('nan'), float('nan'))
            rp = gaze.pupil_right_coords() or (float('nan'), float('nan'))

            est_x = est_y = smooth_x = smooth_y = float('nan')
            if calib_model and gaze.pupils_located and not math.isnan(g_h) and not math.isnan(g_v):
                try:
                    est_x = calib_model['x_slope'] * g_h + calib_model['x_intercept']
                    est_y = calib_model['y_slope'] * g_v + calib_model['y_intercept']
                    hist_x.append(est_x)
                    hist_y.append(est_y)
                    if hist_x and hist_y:
                        smooth_x = float(sum(hist_x) / len(hist_x))
                        smooth_y = float(sum(hist_y) / len(hist_y))
                except Exception:
                    # If model keys missing, skip calibrated values
                    pass

            writer.writerow([
                t, yaw, pitch, roll,
                g_h, g_v, int(blink),
                lp[0], lp[1], rp[0], rp[1],
                est_x, est_y, smooth_x, smooth_y,
            ])

            # fixation dot (if enabled)
            if dot_image is not None:
                cv2.imshow('Fixation Dot', dot_image)

            # live feedback
            annotated = gaze.annotated_frame()
            cv2.imshow('Record (q to quit)', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if end_time and t >= end_time:
                break

    cap.release()
    cv2.destroyAllWindows()
    if dot_image is not None:
        try:
            cv2.destroyWindow('Fixation Dot')
        except cv2.error:
            pass

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', '-o', default='recording.csv')
    p.add_argument('--duration', '-d', type=float, default=0.0,
                   help='duration seconds (0 = until q pressed)')
    p.add_argument('--show-dot', action='store_true',
                   help='Display a fixation dot window during recording.')
    p.add_argument('--dot-size', type=int, default=700,
                   help='Pixel size of the fixation-dot window (square).')
    p.add_argument('--dot-radius', type=int, default=18,
                   help='Radius in pixels of the fixation dot.')
    p.add_argument('--label', type=str, default=None,
                   help='Optional session label to auto-load calibration.')
    p.add_argument('--smooth-window', type=int, default=8,
                   help='Smoothing window size for calibrated gaze.')
    args = p.parse_args()
    main(
        args.out,
        args.duration,
        show_dot=args.show_dot,
        dot_size=args.dot_size,
        dot_radius=args.dot_radius,
        calib_model=None,
        smooth_window=args.smooth_window,
        label=args.label,
    )