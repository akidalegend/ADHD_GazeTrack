"""Automation helper for the sustained-attention fixation task."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError('pandas is required for run_fixation_task.py. Install it via "pip install pandas".') from exc

from analysis import compute_summary
from collect_data import main as collect_main
from task_utils import append_master_row, ensure_dir, prompt_label


def _read_time_bounds(csv_path: Path) -> Tuple[float, float]:
    df = pd.read_csv(csv_path, usecols=['t'])
    if df.empty:
        raise ValueError(f'No timestamps found in {csv_path}')
    return float(df['t'].iloc[0]), float(df['t'].iloc[-1])


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

    collect_main(
        str(raw_path),
        duration,
        show_dot=show_dot,
        dot_size=dot_size,
        dot_radius=dot_radius,
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
