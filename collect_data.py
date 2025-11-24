import time
import csv
import argparse
import math
from pathlib import Path

import cv2  # type: ignore
import numpy as np

from gaze_tracking import GazeTracking

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


def main(output, duration, show_dot=False, dot_size=700, dot_radius=18, dot_color=(0, 0, 255)):
    gaze = GazeTracking()
    cap = cv2.VideoCapture(0)
    start = time.time()
    end_time = start + duration if duration > 0 else None

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
        writer.writerow(['t', 'yaw', 'pitch', 'roll',
                         'g_horizontal', 'is_blinking',
                         'left_px', 'left_py', 'right_px', 'right_py'])

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
            blink = bool(gaze.is_blinking()) if gaze.pupils_located else False

            lp = gaze.pupil_left_coords() or (float('nan'), float('nan'))
            rp = gaze.pupil_right_coords() or (float('nan'), float('nan'))

            writer.writerow([t, yaw, pitch, roll, g_h, int(blink),
                             lp[0], lp[1], rp[0], rp[1]])

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
    args = p.parse_args()
    main(args.out, args.duration, show_dot=args.show_dot,
         dot_size=args.dot_size, dot_radius=args.dot_radius)