"""Antisaccade task runner: show center+target dots while instructing gaze to mirror the target."""
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
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore[attr-defined]
import numpy as np

from analysis import compute_summary
from gaze_tracking import GazeTrackingMediaPipe as GazeTracking
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
    except (KeyError, TypeError, AttributeError):  # pragma: no cover - guardrail
        return float('nan')


def _load_calibration_model(label: str) -> Optional[Dict[str, float]]:
    """Load calibration model for given label. Returns None if not found or invalid."""
    calib_path = Path('sessions/calibration') / f'{label}_calibration.json'
    if not calib_path.exists():
        print(f"No calibration found for '{label}' at {calib_path}. Running without calibration.")
        return None
    try:
        with calib_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        model = data.get('model')
        if model and all(k in model for k in ['x_slope', 'x_intercept', 'y_slope', 'y_intercept']):
            print(f"Loaded calibration model from {calib_path}")
            return model
        else:
            print(f"Invalid calibration model in {calib_path}. Running without calibration.")
            return None
    except (json.JSONDecodeError, OSError) as e:
        print(f"Failed to load calibration from {calib_path}: {e}. Running without calibration.")
        return None


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


def _configure_camera(cap: cv2.VideoCapture, width: int = 640, height: int = 480, fps: int = 30) -> None:
    """Set capture properties to stabilize pupil detection."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def _pupil_detection_check(gaze: GazeTracking, cap: cv2.VideoCapture, samples: int = 40) -> float:
    """Sample a few frames to estimate pupil detection rate before running trials."""
    hits = 0
    total = 0
    for _ in range(samples):
        ret, frame = cap.read()
        if not ret:
            break
        gaze.refresh(frame)
        total += 1
        if gaze.pupils_located:
            hits += 1
    rate = hits / total if total else 0.0
    print(f"Pupil detection warmup: {rate:.0%} ({hits}/{total})")
    return rate


def _wait_for_click(window_name: str, size: int, text: str = 'Click to start') -> bool:
    """Shows a start screen and waits for a left-click or 'q' to cancel."""
    clicked = False

    def _on_mouse(event, _x, _y, _flags, _param):  # pragma: no cover - UI event
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True

    cv2.setMouseCallback(window_name, _on_mouse)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(text, font, 1.2, 3)[0]
        text_x = (size - text_size[0]) // 2
        text_y = (size + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y), font, 1.2, (0, 255, 255), 3)
        cv2.rectangle(canvas, (text_x - 20, text_y - text_size[1] - 20), (text_x + text_size[0] + 20, text_y + 20), (0, 255, 0), 2)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF
        if clicked:
            return True
        if key in (ord('q'), 27):  # q or ESC
            return False


def _run_countdown(cap: cv2.VideoCapture, stimulus_size: int, duration: int = 5) -> bool:
    """
    Displays a countdown on the stimulus window and a 'Get Ready' overlay on the capture window.
    Returns True if the countdown completes, False if the user quits (presses 'q').
    """
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        remaining = math.ceil(duration - elapsed)

        if remaining <= 0:
            break

        # Stimulus window: Black background with white countdown text
        stim_img = np.zeros((stimulus_size, stimulus_size, 3), dtype=np.uint8)
        text = str(remaining)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4.0
        thickness = 10
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (stimulus_size - text_size[0]) // 2
        text_y = (stimulus_size + text_size[1]) // 2
        cv2.putText(stim_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        cv2.imshow('Antisaccade Stimulus', stim_img)

        # Capture window: Live feed with "Get Ready" overlay
        ret, frame = cap.read()
        if ret:
            h = frame.shape[0]
            overlay_text = f"Get Ready: {remaining}"
            cv2.putText(frame, overlay_text, (50, h // 2), font, 1.5, (0, 255, 255), 3)
            cv2.imshow('Antisaccade Capture', frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q or ESC
            return False
    
    return True


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
    slug = f'{label}_antisaccade'
    raw_dir = ensure_dir(raw_dir)
    raw_path = raw_dir / f'{slug}_{timestamp:%Y%m%d_%H%M%S}.csv'

    # Load calibration model (if available)
    calib_model = _load_calibration_model(label)

    # One-Euro filters for calibrated gaze (pixels)
    x_filt = OneEuroFilter(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
    y_filt = OneEuroFilter(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)

    gaze = GazeTracking()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Unable to access camera. Ensure it is connected and free.')

    _configure_camera(cap, width=640, height=480, fps=30)
    warmup_rate = _pupil_detection_check(gaze, cap, samples=40)
    qc_flag = "ok" if warmup_rate >= 0.6 else "poor_pupil_detection"

    cv2.namedWindow('Antisaccade Stimulus', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Antisaccade Stimulus', stimulus_size, stimulus_size)
    cv2.setWindowProperty('Antisaccade Stimulus', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.namedWindow('Antisaccade Capture', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Antisaccade Capture', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Wait for explicit click before starting countdown/trials
    if not _wait_for_click('Antisaccade Stimulus', stimulus_size):
        cap.release()
        cv2.destroyAllWindows()
        print("Task cancelled before start click.")
        return raw_path

    if not _run_countdown(cap, stimulus_size):
        cap.release()
        cv2.destroyAllWindows()
        print("Task cancelled during countdown.")
        return raw_path

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
            'g_horizontal', 'g_vertical', 'is_blinking',
            'left_px', 'left_py', 'right_px', 'right_py',
            'est_gaze_x', 'est_gaze_y', 'smooth_gaze_x', 'smooth_gaze_y',
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
            g_v = gaze.vertical_ratio() if gaze.pupils_located else float('nan')
            blink = bool(gaze.is_blinking()) if gaze.pupils_located else False
            lp = gaze.pupil_left_coords() or (float('nan'), float('nan'))
            rp = gaze.pupil_right_coords() or (float('nan'), float('nan'))

            # Apply calibration mapping and smoothing if model is available
            est_x = est_y = smooth_x = smooth_y = float('nan')
            if calib_model and gaze.pupils_located and g_h is not None and g_v is not None:
                if not math.isnan(g_h) and not math.isnan(g_v):
                    est_x = calib_model['x_slope'] * g_h + calib_model['x_intercept']
                    est_y = calib_model['y_slope'] * g_v + calib_model['y_intercept']
                    smooth_x = x_filt.update(now, est_x)
                    smooth_y = y_filt.update(now, est_y)

            writer.writerow([
                now, yaw, pitch, roll,
                g_h, g_v, int(blink),
                lp[0], lp[1], rp[0], rp[1],
                est_x, est_y, smooth_x, smooth_y,
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
            cv2.imshow('Antisaccade Stimulus', stim_img)
            annotated = gaze.annotated_frame()
            cv2.imshow('Antisaccade Capture', annotated)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q or ESC
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
        'task': 'antisaccade',
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
                'time': event.onset_time,
                'x': event.target_center[0],
                'y': event.target_center[1],
                'fixation_x': event.fixation_center[0],
                'fixation_y': event.fixation_center[1],
            }
            for event in events
        ],
        'qc_pupil_warmup_rate': warmup_rate,
        'qc_flag': qc_flag,
    }
    summary_with_meta = {**metadata, **summary}

    if summary_dir:
        summary_dir = ensure_dir(summary_dir)
        summary_path = summary_dir / f'{slug}_{timestamp:%Y%m%d_%H%M%S}.json'
        summary_path.write_text(json.dumps(summary_with_meta, indent=2), encoding='utf-8')

    append_master_row(master_csv, metadata, summary)
    return raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Antisaccade task: look opposite the target while matching prosaccade timing.'
    )
    parser.add_argument('--label', help='Participant/session label (will prompt if omitted).')
    parser.add_argument('--trials', type=int, default=20, help='Number of antisaccade trials to run.')
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
    print("\nAntisaccade task instructions:")
    print("- Two windows open fullscreen: Stimulus (targets) and Capture (camera).")
    print("- Click the Stimulus window to start; q/Esc cancels.")
    print("- During trials: look opposite the green target; press q/Esc anytime to stop.\n")
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
    print(
        'Antisaccade task complete. Raw session saved at '
        f'{raw_path}. Metrics appended to {args.master_csv}.'
    )


# --- One Euro filter (for calibrated gaze smoothing) ---
class _LowPass:
    def __init__(self) -> None:
        self._y: float | None = None

    def reset(self) -> None:
        self._y = None

    def apply(self, x: float, a: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = a * x + (1.0 - a) * self._y
        return self._y


class OneEuroFilter:
    def __init__(self, *, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._t_prev: float | None = None
        self._x_prev: float | None = None
        self._dx = _LowPass()
        self._x = _LowPass()

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        if dt <= 0:
            return 1.0
        r = 2.0 * math.pi * float(cutoff) * dt
        return r / (r + 1.0)

    def reset(self) -> None:
        self._t_prev = None
        self._x_prev = None
        self._dx.reset()
        self._x.reset()

    def update(self, t: float, x: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = float(t)
            self._x_prev = float(x)
            return self._x.apply(float(x), 1.0)

        dt = float(t) - float(self._t_prev)
        self._t_prev = float(t)

        dx = (float(x) - float(self._x_prev)) / dt if dt > 0
