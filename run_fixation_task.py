"""Automation helper for the sustained-attention fixation task."""
from __future__ import annotations

import argparse
import json
import math
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError('pandas is required for run_fixation_task.py. Install it via "pip install pandas".') from exc

from analysis import compute_summary, load_calibration_model
from collect_data import main as collect_main
from task_utils import append_master_row, ensure_dir, prompt_label


def _read_time_bounds(csv_path: Path) -> Tuple[float, float]:
    df = pd.read_csv(csv_path, usecols=['t'])
    if df.empty:
        raise ValueError(f'No timestamps found in {csv_path}')
    return float(df['t'].iloc[0]), float(df['t'].iloc[-1])


def _run_countdown(duration=5):
    """Opens camera briefly to show countdown before main task starts."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access camera.")
        return False
    window_name = "Get Ready"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        remaining = math.ceil(duration - elapsed)
        if remaining <= 0:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, f"Starting in {remaining}s", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q or ESC
            cap.release()
            cv2.destroyAllWindows()
            return False
            
    cap.release()
    cv2.destroyAllWindows()
    return True


def _wait_for_click(window_name: str = "Fixation Start", width: int = 500, height: int = 300, text: str = "Click to start") -> bool:
    """Shows a start screen and waits for a left-click or 'q' to cancel."""
    clicked = False

    def _on_mouse(event, _x, _y, _flags, _param):  # pragma: no cover - UI event
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setMouseCallback(window_name, _on_mouse)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y), font, 1.0, (0, 255, 255), 2)
        cv2.rectangle(canvas, (text_x - 20, text_y - text_size[1] - 20), (text_x + text_size[0] + 20, text_y + 20), (0, 255, 0), 2)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF
        if clicked:
            cv2.destroyWindow(window_name)
            return True
        if key in (ord('q'), 27):  # q or ESC
            cv2.destroyWindow(window_name)
            return False


def run_fixation_task(
    *,
    label: Optional[str],
    duration: float,
    master_csv: Path,
    raw_dir: Path,
    summary_dir: Optional[Path],
    show_dot: bool,
    dot_size: int,
    dot_radius: int,
) -> Path:
    timestamp = datetime.now()
    slug = label or timestamp.strftime('session_%Y%m%d_%H%M%S')
    raw_dir = ensure_dir(raw_dir)
    raw_path = raw_dir / f'{slug}_{timestamp:%Y%m%d_%H%M%S}.csv'

    if not _wait_for_click():
        print("Task cancelled before start click.")
        return raw_path

    # --- INSERT COUNTDOWN HERE ---
    if not _run_countdown():
        print("Task cancelled.")
        return raw_path
    # -----------------------------

    # Try to load calibration model for this label if provided
    calib_model = load_calibration_model(label) if label else None

    collect_main(
        str(raw_path),
        duration,
        show_dot=show_dot,
        dot_size=dot_size,
        dot_radius=dot_radius,
        calib_model=calib_model,
        label=label,
    )

    if not raw_path.exists():
        raise FileNotFoundError(f'Expected raw session at {raw_path} but it was not created.')

    start_t, end_t = _read_time_bounds(raw_path)
    summary = compute_summary(
        str(raw_path),
        interval_windows=[(start_t, end_t)],
    )
    metadata = {
        'timestamp_iso': timestamp.isoformat(),
        'session_label': slug,
        'task': 'fixation_dot',
        'raw_csv': str(raw_path),
        'configured_duration_s': duration,
        'configured_trials': None,
        'center_duration_s': None,
        'gap_duration_s': None,
        'target_duration_s': None,
        'stimuli_directions': None,
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
        description='Guide participant through fixation task, capture data, and update master metrics.'
    )
    parser.add_argument('--label', help='Optional session label / participant ID.')
    parser.add_argument('--duration', type=float, default=25.0, help='Recording duration in seconds.')
    parser.add_argument('--master-csv', default='master_metrics.csv', help='Path to cumulative metrics CSV.')
    parser.add_argument('--raw-dir', default='sessions/raw', help='Directory to store raw per-frame CSV files.')
    parser.add_argument('--summary-dir', default='sessions/summaries', help='Directory for per-session JSON summaries.')
    parser.add_argument('--no-dot', action='store_true', help='Disable fixation-dot window (manual stimulus).')
    parser.add_argument('--dot-size', type=int, default=700, help='Pixel size of fixation-dot window.')
    parser.add_argument('--dot-radius', type=int, default=18, help='Radius of fixation dot (pixels).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_label = prompt_label(args.label)
    master_csv = Path(args.master_csv)
    raw_dir = Path(args.raw_dir)
    summary_dir = Path(args.summary_dir) if args.summary_dir else None

    raw_path = run_fixation_task(
        label=resolved_label,
        duration=args.duration,
        master_csv=master_csv,
        raw_dir=raw_dir,
        summary_dir=summary_dir,
        show_dot=not args.no_dot,
        dot_size=args.dot_size,
        dot_radius=args.dot_radius,
    )
    print(f'Fixation task complete. Raw session saved at {raw_path}. Metrics appended to {master_csv}.')


if __name__ == '__main__':
    main()
