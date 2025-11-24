"""Prosaccade task runner: shows center + peripheral targets, records gaze data, logs metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # type: ignore[attr-defined]
import numpy as np

from analysis import compute_summary
from gaze_tracking import GazeTracking
from task_utils import append_master_row, ensure_dir, prompt_label


@dataclass
class StimulusEvent:
    trial: int
    angle_deg: float
    target_center: Tuple[int, int]
    fixation_center: Tuple[int, int]
    onset_time: float


def safe_angle(head_pose, key: str) -> float:
    try:
        return float(head_pose['angles'].get(key))
    except (KeyError, TypeError, AttributeError):  # pragma: no cover - fallback safety
        return float('nan')


def safe_pupil(coords, idx: int) -> float:
    try:
        return float(coords[idx])
    except (IndexError, TypeError):  # pragma: no cover - fallback safety
        return float('nan')


def _stimulus_image(
    state: str,
    size: int,
    fixation_center: Tuple[int, int],
    center_radius: int,
    target_radius: int,
    target_center: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if state == 'center':
        cv2.circle(canvas, fixation_center, center_radius, (0, 0, 255), -1)
    elif state == 'target' and target_center is not None:
        cv2.circle(canvas, target_center, target_radius, (0, 255, 0), -1)
    return canvas


def _random_point(size: int, margin: int) -> Tuple[int, int]:
    lower = margin
    upper = size - margin
    return (
        random.randint(lower, upper),
        random.randint(lower, upper),
    )


def run_trials(
    *,
    label: str,
    trials: int,
    center_duration: float,
    gap_duration: float,
    target_duration: float,
    stimulus_size: int,
    center_radius: int,
    target_radius: int,
    offset_ratio: float,
    master_csv: Path,
    raw_dir: Path,
    summary_dir: Optional[Path],
) -> Path:
    timestamp = datetime.now()
    slug = f'{label}_prosaccade'
    raw_dir = ensure_dir(raw_dir)
    raw_path = raw_dir / f'{slug}_{timestamp:%Y%m%d_%H%M%S}.csv'

    gaze = GazeTracking()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Unable to access camera. Ensure it is connected and not used elsewhere.')

    cv2.namedWindow('Prosaccade Stimulus', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Prosaccade Stimulus', stimulus_size, stimulus_size)

    cv2.namedWindow('Prosaccade Capture', cv2.WINDOW_NORMAL)

    margin_center = max(center_radius, 5)
    margin_target = max(target_radius, 5)

    events: List[StimulusEvent] = []
    trial_idx = 0
    state = 'center'
    current_target_center: Optional[Tuple[int, int]] = None
    current_angle_deg: Optional[float] = None
    current_fixation_center: Tuple[int, int] = _random_point(stimulus_size, margin_center)
    phase_end = time.time() + center_duration

    with raw_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            't', 'yaw', 'pitch', 'roll',
            'g_horizontal', 'is_blinking',
            'left_px', 'left_py', 'right_px', 'right_py',
            'trial_index', 'stimulus_state', 'stimulus_angle_deg',
            'stimulus_x', 'stimulus_y', 'fixation_x', 'fixation_y'
        ])

        while trial_idx < trials:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            if now >= phase_end:
                if state == 'center':
                    state = 'gap'
                    phase_end = now + gap_duration
                elif state == 'gap':
                    state = 'target'
                    angle_rad = random.uniform(0.0, math.tau)
                    offset = int(stimulus_size * offset_ratio)
                    target_x = int(current_fixation_center[0] + math.cos(angle_rad) * offset)
                    target_y = int(current_fixation_center[1] + math.sin(angle_rad) * offset)
                    target_x = max(margin_target, min(stimulus_size - margin_target, target_x))
                    target_y = max(margin_target, min(stimulus_size - margin_target, target_y))
                    current_target_center = (target_x, target_y)
                    current_angle_deg = math.degrees(angle_rad) % 360.0
                    phase_end = now + target_duration
                    events.append(
                        StimulusEvent(
                            trial=trial_idx,
                            angle_deg=current_angle_deg,
                            target_center=current_target_center,
                            fixation_center=current_fixation_center,
                            onset_time=now,
                        )
                    )
                elif state == 'target':
                    trial_idx += 1
                    state = 'center'
                    current_target_center = None
                    current_angle_deg = None
                    current_fixation_center = _random_point(stimulus_size, margin_center)
                    phase_end = now + center_duration

            gaze.refresh(frame)
            hp = gaze.head_pose or {}
            yaw = safe_angle(hp, 'yaw')
            pitch = safe_angle(hp, 'pitch')
            roll = safe_angle(hp, 'roll')
            g_h = gaze.horizontal_ratio() if gaze.pupils_located else float('nan')
            blink = bool(gaze.is_blinking()) if gaze.pupils_located else False
            lp = gaze.pupil_left_coords() or (float('nan'), float('nan'))
            rp = gaze.pupil_right_coords() or (float('nan'), float('nan'))

            writer.writerow([
                now, yaw, pitch, roll,
                g_h, int(blink),
                lp[0], lp[1], rp[0], rp[1],
                trial_idx, state,
                current_angle_deg if current_angle_deg is not None else '',
                current_target_center[0] if current_target_center else '',
                current_target_center[1] if current_target_center else '',
                current_fixation_center[0],
                current_fixation_center[1],
            ])

            stim_img = _stimulus_image(
                state,
                stimulus_size,
                current_fixation_center,
                center_radius,
                target_radius,
                current_target_center,
            )
            cv2.imshow('Prosaccade Stimulus', stim_img)
            annotated = gaze.annotated_frame()
            cv2.imshow('Prosaccade Capture', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if not raw_path.exists():
        raise FileNotFoundError(f'Expected raw session at {raw_path}')

    stimuli_times = [event.onset_time for event in events]
    summary = compute_summary(
        str(raw_path),
        stimuli_times=stimuli_times,
    )

    metadata = {
        'timestamp_iso': timestamp.isoformat(),
        'session_label': label,
        'task': 'prosaccade',
        'raw_csv': str(raw_path),
        'configured_duration_s': None,
        'configured_trials': trials,
        'center_duration_s': center_duration,
        'gap_duration_s': gap_duration,
        'target_duration_s': target_duration,
        'stimuli_directions': [
            {
                'trial': event.trial,
                'angle_deg': event.angle_deg,
                'x': event.target_center[0],
                'y': event.target_center[1],
                'fixation_x': event.fixation_center[0],
                'fixation_y': event.fixation_center[1],
            }
            for event in events
        ],
    }
    summary_with_meta = {**metadata, **summary}

    if summary_dir:
        summary_dir = ensure_dir(summary_dir)
        summary_path = summary_dir / f'{slug}_{timestamp:%Y%m%d_%H%M%S}.json'
        summary_path.write_text(json.dumps(summary_with_meta, indent=2), encoding='utf-8')

    append_master_row(master_csv, metadata, summary)
    return raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prosaccade task runner (baseline motor control).')
    parser.add_argument('--label', help='Participant/session label (will prompt if omitted).')
    parser.add_argument('--trials', type=int, default=20, help='Number of prosaccade trials to run.')
    parser.add_argument('--center-duration', type=float, default=1.0, help='Seconds to show center fixation per trial.')
    parser.add_argument('--gap-duration', type=float, default=0.2, help='Gap (seconds) between center offset and target onset.')
    parser.add_argument('--target-duration', type=float, default=1.5, help='Seconds to keep target visible.')
    parser.add_argument('--stimulus-size', type=int, default=600, help='Stimulus window size in pixels.')
    parser.add_argument('--center-radius', type=int, default=12, help='Radius of center fixation dot (pixels).')
    parser.add_argument('--target-radius', type=int, default=14, help='Radius of peripheral target (pixels).')
    parser.add_argument('--offset-ratio', type=float, default=0.35, help='Offset from center as ratio of window width (0-0.5).')
    parser.add_argument('--master-csv', default='master_metrics.csv', help='Path to cumulative metrics CSV.')
    parser.add_argument('--raw-dir', default='sessions/raw', help='Directory to store raw per-frame CSV files.')
    parser.add_argument('--summary-dir', default='sessions/summaries', help='Directory for per-session JSON summaries.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = prompt_label(args.label)
    raw_path = run_trials(
        label=label,
        trials=args.trials,
        center_duration=args.center_duration,
        gap_duration=args.gap_duration,
        target_duration=args.target_duration,
        stimulus_size=args.stimulus_size,
        center_radius=args.center_radius,
        target_radius=args.target_radius,
        offset_ratio=args.offset_ratio,
        master_csv=Path(args.master_csv),
        raw_dir=Path(args.raw_dir),
        summary_dir=Path(args.summary_dir) if args.summary_dir else None,
    )
    print(f'Prosaccade task complete. Raw session saved at {raw_path}. Metrics appended to {args.master_csv}.')


if __name__ == '__main__':
    main()
