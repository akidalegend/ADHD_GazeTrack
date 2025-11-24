import math
import json
import csv
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        'pandas is required for analysis.py. Install it via "pip install pandas" inside your active environment.'
    ) from exc
import numpy as np

from gaze_tracking.saccades import (
    detect_saccades,
    detect_fixations,
    saccade_latency_to_stimuli,
    count_intrusive_saccades,
)

def load_csv(path):
    df = pd.read_csv(path)
    # ensure sorted by time
    df = df.sort_values('t').reset_index(drop=True)
    return df

def angle_path_length(df):
    a = df[['yaw','pitch','roll']].ffill().fillna(0).to_numpy()
    diffs = np.linalg.norm(np.diff(a, axis=0), axis=1)
    return float(np.nansum(diffs)), diffs

def angular_speed_stats(diffs, times):
    dt = np.diff(times)
    dt[dt == 0] = 1e-6
    speeds = diffs / dt
    return {
        'mean_speed': float(np.nanmean(speeds)),
        'median_speed': float(np.nanmedian(speeds)),
        'max_speed': float(np.nanmax(speeds)),
        'speeds': speeds
    }

def compute_features(csv_path, spike_threshold=30.0, motion_threshold=5.0, df=None):
    df = load_csv(csv_path) if df is None else df
    duration = df['t'].iloc[-1] - df['t'].iloc[0] if len(df) > 1 else 0.0

    path_len, diffs = angle_path_length(df)
    times = df['t'].to_numpy()
    speed_stats = angular_speed_stats(diffs, times) if len(diffs)>0 else {'mean_speed':0.0,'speeds':np.array([])}

    speeds = speed_stats['speeds']
    spikes = int(np.sum(speeds > spike_threshold)) if speeds.size else 0
    percent_time_moving = float(np.sum(speeds > motion_threshold) / (len(speeds)) ) if speeds.size else 0.0

    blink_rate = float(df['is_blinking'].sum()) / (duration+1e-6) if 'is_blinking' in df else 0.0
    gaze_dispersion = float(df['g_horizontal'].std(skipna=True)) if 'g_horizontal' in df else float('nan')

    features = {
        'duration_s': duration,
        'path_length_deg': path_len,
        'mean_angular_speed_deg_per_s': speed_stats['mean_speed'],
        'spike_count': spikes,
        'percent_time_moving': percent_time_moving,
        'blink_rate_per_s': blink_rate,
        'gaze_dispersion': gaze_dispersion
    }
    return features


def _select_gaze_series(df):
    times = pd.to_numeric(df['t'], errors='coerce').to_numpy(dtype=float)
    candidate_cols = ['g_horizontal', 'left_px', 'right_px']
    series = None
    for col in candidate_cols:
        if col in df.columns:
            col_values = pd.to_numeric(df[col], errors='coerce')
            if col_values.notna().sum() > 0:
                series = col_values.to_numpy(dtype=float)
                break
    if series is None:
        series = np.zeros_like(times)
    return times, series


def _clean_numeric(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [_clean_numeric(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_numeric(v) for k, v in value.items()}
    return value


def compute_saccade_metrics(
    csv_path,
    df=None,
    vel_thresh=0.8,
    min_dur=0.015,
    smooth_w=5,
    min_fix_dur=0.08,
    stimuli_times=None,
    interval_windows=None,
    latency_window=1.0,
):
    df = load_csv(csv_path) if df is None else df
    times, pos = _select_gaze_series(df)
    saccades = detect_saccades(times, pos, vel_thresh=vel_thresh, min_dur=min_dur, smooth_w=smooth_w)
    fixations = detect_fixations(times, pos, saccades, min_fix_dur=min_fix_dur)

    latencies = []
    if stimuli_times:
        latencies = saccade_latency_to_stimuli(saccades, stimuli_times, max_latency=latency_window)
        latencies = [lat if math.isfinite(lat) else None for lat in latencies]

    intrusive_total, intrusive_breakdown = (0, [])
    if interval_windows:
        intrusive_total, intrusive_breakdown = count_intrusive_saccades(saccades, interval_windows)

    metrics = {
        'saccade_count': len(saccades),
        'fixation_count': len(fixations),
        'mean_saccade_duration_s': float(np.mean([s['duration'] for s in saccades])) if saccades else 0.0,
        'median_saccade_duration_s': float(np.median([s['duration'] for s in saccades])) if saccades else 0.0,
        'mean_saccade_peak_velocity': float(np.mean([s['peak_velocity'] for s in saccades])) if saccades else 0.0,
        'mean_saccade_amplitude': float(np.mean([s['amplitude'] for s in saccades])) if saccades else 0.0,
        'mean_fixation_duration_s': float(np.mean([f['duration'] for f in fixations])) if fixations else 0.0,
        'saccade_latencies_s': latencies,
        'intrusive_saccade_count': intrusive_total,
        'intrusive_counts_per_interval': intrusive_breakdown,
    }
    return metrics


def compute_summary(
    csv_path,
    spike_threshold=30.0,
    motion_threshold=5.0,
    vel_thresh=0.8,
    min_dur=0.015,
    smooth_w=5,
    min_fix_dur=0.08,
    stimuli_times=None,
    interval_windows=None,
    latency_window=1.0,
):
    df = load_csv(csv_path)
    combined = compute_features(
        csv_path,
        spike_threshold=spike_threshold,
        motion_threshold=motion_threshold,
        df=df,
    )
    saccade_metrics = compute_saccade_metrics(
        csv_path,
        df=df,
        vel_thresh=vel_thresh,
        min_dur=min_dur,
        smooth_w=smooth_w,
        min_fix_dur=min_fix_dur,
        stimuli_times=stimuli_times,
        interval_windows=interval_windows,
        latency_window=latency_window,
    )
    combined.update(saccade_metrics)
    return _clean_numeric(combined)


def _load_scalar_list(path):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [float(item) for item in data]
    except json.JSONDecodeError:
        pass
    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # allow comma-separated values per line but only take the first entry
        token = line.split(',')[0].strip()
        values.append(float(token))
    return values


def _load_interval_list(path):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        intervals = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and 'start' in entry and 'end' in entry:
                    intervals.append((float(entry['start']), float(entry['end'])))
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    intervals.append((float(entry[0]), float(entry[1])))
        if intervals:
            return intervals
    except json.JSONDecodeError:
        pass
    parsed_intervals = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(';', ',').split(',') if p.strip()]
        if len(parts) >= 2:
            parsed_intervals.append((float(parts[0]), float(parts[1])))
    return parsed_intervals


def _append_dict_to_csv(path, data):
    path = Path(path)
    write_header = not path.exists()
    # preserve insertion order for readability
    fieldnames = list(data.keys())
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute gaze/head metrics and saccade statistics for a session CSV.')
    parser.add_argument('csv_path', help='Path to session CSV produced by collect_data.py')
    parser.add_argument('--spike-threshold', type=float, default=30.0, help='Angular speed threshold for counting spikes (deg/s).')
    parser.add_argument('--motion-threshold', type=float, default=5.0, help='Angular speed threshold for percent_time_moving (deg/s).')
    parser.add_argument('--vel-thresh', type=float, default=0.8, help='Velocity threshold for saccade detection (gaze-units/s).')
    parser.add_argument('--min-saccade-dur', type=float, default=0.015, help='Minimum saccade duration in seconds.')
    parser.add_argument('--smooth-window', type=int, default=5, help='Window size for moving-average smoothing before velocity calc.')
    parser.add_argument('--min-fix-dur', type=float, default=0.08, help='Minimum fixation duration in seconds.')
    parser.add_argument('--latency-window', type=float, default=1.0, help='Maximum latency window (s) when pairing stimuli to saccades.')
    parser.add_argument('--stimuli-file', help='Optional path to JSON or newline file listing stimulus onset timestamps (seconds).')
    parser.add_argument('--intervals-file', help='Optional path to JSON or newline file listing intrusive-interval pairs (start,end).')
    parser.add_argument('--out', help='Write summary JSON to this path.')
    parser.add_argument('--csv-out', help='Append the summary as a CSV row at this path.')

    args = parser.parse_args()

    stimuli_time_values = _load_scalar_list(args.stimuli_file) if args.stimuli_file else None
    interval_window_values = _load_interval_list(args.intervals_file) if args.intervals_file else None

    summary_payload = compute_summary(
        args.csv_path,
        spike_threshold=args.spike_threshold,
        motion_threshold=args.motion_threshold,
        vel_thresh=args.vel_thresh,
        min_dur=args.min_saccade_dur,
        smooth_w=args.smooth_window,
        min_fix_dur=args.min_fix_dur,
        stimuli_times=stimuli_time_values,
        interval_windows=interval_window_values,
        latency_window=args.latency_window,
    )

    print(json.dumps(summary_payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
    if args.csv_out:
        _append_dict_to_csv(args.csv_out, summary_payload)