"""Shared utilities for experimental task scripts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

MASTER_FIELDS = [
    'timestamp_iso',
    'session_label',
    'task',
    'raw_csv',
    'configured_duration_s',
    'configured_trials',
    'center_duration_s',
    'gap_duration_s',
    'target_duration_s',
    'duration_s',
    'path_length_deg',
    'mean_angular_speed_deg_per_s',
    'spike_count',
    'percent_time_moving',
    'blink_rate_per_s',
    'gaze_dispersion',
    'saccade_count',
    'fixation_count',
    'mean_saccade_duration_s',
    'median_saccade_duration_s',
    'mean_saccade_peak_velocity',
    'mean_saccade_amplitude',
    'mean_fixation_duration_s',
    'saccade_latencies_s',
    'intrusive_saccade_count',
    'intrusive_counts_per_interval',
    'stimuli_directions',
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directories(paths: Union[str, List[str]]) -> None:
    """Ensures directories exist (String/List version for compatibility)."""
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def prompt_label(default_label: Optional[str] = None) -> str:
    prompt = 'Enter session label'
    if default_label:
        prompt += f' [{default_label}]'
    prompt += ': '
    while True:
        user_input = input(prompt).strip()
        if user_input:
            return user_input
        if default_label:
            return default_label
        print('Session label is required. Please enter a non-empty value.')


def _serialize_value(value: Any) -> Any:
    if value is None:
        return ''
    if isinstance(value, float) and (value != value):  # NaN check
        return ''
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def append_master_row(master_csv: Path, metadata: Dict[str, Any], summary: Dict[str, Any]) -> None:
    row = {}
    combined = {**summary, **metadata}
    for field in MASTER_FIELDS:
        row[field] = _serialize_value(combined.get(field))

    write_header = not master_csv.exists()
    with master_csv.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MASTER_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
