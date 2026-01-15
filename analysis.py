import math
import json
import csv
import os
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


def _infer_task_from_path(csv_path: str) -> str | None:
    name = os.path.basename(str(csv_path)).lower()
    if 'antisaccade' in name:
        return 'antisaccade'
    if 'prosaccade' in name:
        return 'prosaccade'
    return None


def _extract_target_onsets_and_expected_dirs(df):
    """
    Extract target-onset times and expected LEFT/RIGHT direction from the task CSV.

    Expected direction is derived from stimulus_x relative to fixation_x at the target-onset row.
    Returns list[(onset_time, expected_dir)] where expected_dir is -1 (left), +1 (right), or None.
    """
    required = {'t', 'stimulus_state', 'stimulus_x', 'fixation_x'}
    if not required.issubset(set(df.columns)):
        return []

    is_target = df['stimulus_state'] == 'target'
    transitions = is_target & (~is_target.shift(1).fillna(False))
    if not transitions.any():
        return []

    rows = df.loc[transitions, ['t', 'stimulus_x', 'fixation_x']].copy()
    rows['t'] = pd.to_numeric(rows['t'], errors='coerce')
    rows['stimulus_x'] = pd.to_numeric(rows['stimulus_x'], errors='coerce')
    rows['fixation_x'] = pd.to_numeric(rows['fixation_x'], errors='coerce')

    out = []
    for _, r in rows.iterrows():
        onset_t = float(r['t']) if pd.notna(r['t']) else None
        if onset_t is None:
            continue
        expected = None
        if pd.notna(r['stimulus_x']) and pd.notna(r['fixation_x']):
            dx = float(r['stimulus_x']) - float(r['fixation_x'])
            if abs(dx) >= 1e-6:
                expected = 1 if dx > 0 else -1
        out.append((onset_t, expected))
    return out


def _select_direction_signal(df):
    """Return (times, pos) for direction/latency analysis.

    We prefer the raw `g_horizontal` ratio because it is monotonic with horizontal gaze and
    does not depend on screen calibration. We convert it to a signed signal where
    RIGHT is positive and LEFT is negative.

    Note: In this codebase, `g_horizontal` is clamped to [0,1] and tends to be smaller
    when looking right and larger when looking left (see gaze_tracking_mediapipe.py).
    """
    if 't' not in df.columns or 'g_horizontal' not in df.columns:
        return None

    times = pd.to_numeric(df['t'], errors='coerce').to_numpy(dtype=float)
    gh = pd.to_numeric(df['g_horizontal'], errors='coerce').to_numpy(dtype=float)

    # Convert to signed: right positive, left negative
    pos = 0.5 - gh
    return times, pos


def _first_saccade_after(saccades, onset_t: float, max_latency: float):
    if not saccades:
        return None
    for s in saccades:
        if s['onset_t'] >= onset_t and (s['onset_t'] - onset_t) <= max_latency:
            return s
    return None


def compute_reaction_direction_metrics(
    *,
    csv_path: str,
    df,
    vel_thresh: float,
    min_dur: float,
    smooth_w: int,
    latency_window: float,
    amp_thresh: float = 0.03,
    correction_window: float = 0.6,
):
    """Compute trial-level timing+direction metrics for pro/anti-saccade tasks.

    This intentionally prioritizes *timing* and *direction* over absolute gaze coordinates.

    - Timing: latency from target onset to first detected saccade onset.
    - Direction: sign of the first saccade amplitude in the signed horizontal signal.
    - Correctness: depends on task type inferred from filename.
    """

    task = _infer_task_from_path(csv_path)
    if task not in ('prosaccade', 'antisaccade'):
        return {}

    onset_and_dirs = _extract_target_onsets_and_expected_dirs(df)
    if not onset_and_dirs:
        return {}

    selected = _select_direction_signal(df)
    if selected is None:
        return {}
    times, pos = selected

    # Detect saccades in the signed ratio signal.
    # Threshold defaults are in "ratio units per second"; you will likely tune vel_thresh per camera FPS.
    saccades = detect_saccades(times, pos, vel_thresh=vel_thresh, min_dur=min_dur, smooth_w=smooth_w)

    trial_latencies = []
    trial_expected_dirs = []
    trial_observed_dirs = []
    trial_is_correct = []
    trial_is_error_toward_target = []
    trial_has_correction = []

    for onset_t, expected_dir in onset_and_dirs:
        s = _first_saccade_after(saccades, onset_t, latency_window)
        if s is None or expected_dir is None:
            trial_latencies.append(None)
            trial_expected_dirs.append(expected_dir)
            trial_observed_dirs.append(None)
            trial_is_correct.append(None)
            trial_is_error_toward_target.append(None)
            trial_has_correction.append(None)
            continue

        latency = float(s['onset_t'] - onset_t)
        amp = float(s.get('amplitude', 0.0))
        obs_dir = None
        if abs(amp) >= float(amp_thresh):
            obs_dir = 1 if amp > 0 else -1

        trial_latencies.append(latency if math.isfinite(latency) else None)
        trial_expected_dirs.append(int(expected_dir))
        trial_observed_dirs.append(obs_dir)

        if obs_dir is None:
            trial_is_correct.append(None)
            trial_is_error_toward_target.append(None)
            trial_has_correction.append(None)
            continue

        if task == 'prosaccade':
            correct = (obs_dir == expected_dir)
            error_toward = (obs_dir != expected_dir)
        else:  # antisaccade
            correct = (obs_dir == -expected_dir)
            error_toward = (obs_dir == expected_dir)

        trial_is_correct.append(bool(correct))
        trial_is_error_toward_target.append(bool(error_toward))

        # Correction: after an antisaccade "toward" error, do we see a subsequent saccade
        # in the opposite direction shortly after?
        if task == 'antisaccade' and error_toward:
            corr = False
            window_end = float(s['onset_t']) + float(correction_window)
            for s2 in saccades:
                if s2['onset_t'] <= s['onset_t']:
                    continue
                if s2['onset_t'] > window_end:
                    break
                amp2 = float(s2.get('amplitude', 0.0))
                if abs(amp2) < float(amp_thresh):
                    continue
                obs2 = 1 if amp2 > 0 else -1
                if obs2 == -expected_dir:
                    corr = True
                    break
            trial_has_correction.append(bool(corr))
        else:
            trial_has_correction.append(None)

    # Aggregate (ignore None)
    valid_correct_flags = [v for v in trial_is_correct if isinstance(v, bool)]
    valid_latencies = [v for v in trial_latencies if isinstance(v, (int, float))]
    accuracy = float(np.mean(valid_correct_flags)) if valid_correct_flags else None
    median_latency = float(np.median(valid_latencies)) if valid_latencies else None
    mean_latency = float(np.mean(valid_latencies)) if valid_latencies else None

    antisaccade_error_flags = [v for v in trial_is_error_toward_target if isinstance(v, bool)]
    antisaccade_error_rate = float(np.mean(antisaccade_error_flags)) if (task == 'antisaccade' and antisaccade_error_flags) else None
    correction_flags = [v for v in trial_has_correction if isinstance(v, bool)]
    correction_rate = float(np.mean(correction_flags)) if (task == 'antisaccade' and correction_flags) else None

    return {
        'task_inferred': task,
        'trial_latency_s': trial_latencies,
        'trial_expected_dir': trial_expected_dirs,
        'trial_observed_dir': trial_observed_dirs,
        'trial_correct': trial_is_correct,
        'direction_accuracy': accuracy,
        'latency_median_s': median_latency,
        'latency_mean_s': mean_latency,
        'antisaccade_error_rate': antisaccade_error_rate,
        'antisaccade_correction_rate': correction_rate,
        'direction_amp_thresh': float(amp_thresh),
    }

def load_calibration_model(label, calibration_dir="sessions/calibration"):
    """
    Tries to load a calibration JSON for the given label.
    Returns the model dict or None.
    """
    filename = os.path.join(calibration_dir, f"{label}_calibration.json")
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            return data.get("model")
        except Exception as e:
            print(f"Warning: Failed to load calibration file {filename}: {e}")
    return None

def apply_calibration(df, model):
    """
    Applies linear calibration model to gaze data.
    Adds 'gaze_x_px' column to dataframe.
    """
    if model and 'g_horizontal' in df.columns:
        # x_screen = slope * gaze_h + intercept
        slope = model.get("x_slope", 1.0)
        intercept = model.get("x_intercept", 0.0)
        
        # Apply model
        df['gaze_x_px'] = df['g_horizontal'] * slope + intercept
        return True
    return False

def load_csv(path):
    df = pd.read_csv(path)
    # ensure sorted by time
    df = df.sort_values('t').reset_index(drop=True)
    # keep continuity without injecting fake zeros
    df = df.ffill().bfill()
    return df

def _load_csv_raw(path):
    """Raw CSV load (preserve NaNs/empty cells) for QC computations."""
    df = pd.read_csv(path)
    df = df.sort_values('t').reset_index(drop=True)
    return df

def _qc_metrics_and_flag(raw_df):
    """
    Compute QC metrics from raw per-frame CSV.
    Returns (qc_metrics_dict, qc_flag_str, qc_reasons_list).
    """
    qc = {}
    reasons = []

    if raw_df is None or raw_df.empty or 't' not in raw_df.columns:
        return {'qc_valid': False}, 'invalid', ['empty_or_missing_t']

    times = pd.to_numeric(raw_df['t'], errors='coerce').to_numpy(dtype=float)
    duration = float(np.nanmax(times) - np.nanmin(times)) if np.isfinite(times).any() else 0.0
    qc['qc_valid'] = True
    qc['qc_duration_s'] = duration

    if 'is_blinking' in raw_df.columns:
        blink = pd.to_numeric(raw_df['is_blinking'], errors='coerce').fillna(0).to_numpy(dtype=float)
        blink_frames = int(np.nansum(blink > 0))
        total_frames = int(np.sum(np.isfinite(times)))
        qc['qc_blink_fraction'] = float(blink_frames / max(1, total_frames))
    else:
        qc['qc_blink_fraction'] = None

    # "Pupil detection" proxy: gaze ratios present (not NaN)
    if 'g_horizontal' in raw_df.columns and 'g_vertical' in raw_df.columns:
        gh = pd.to_numeric(raw_df['g_horizontal'], errors='coerce')
        gv = pd.to_numeric(raw_df['g_vertical'], errors='coerce')
        valid = (gh.notna() & gv.notna())
        qc['qc_pupil_detection_rate'] = float(valid.mean()) if len(valid) else 0.0
    else:
        qc['qc_pupil_detection_rate'] = None

    # Calibrated gaze availability (from task runners)
    if 'smooth_gaze_x' in raw_df.columns:
        sx = pd.to_numeric(raw_df['smooth_gaze_x'], errors='coerce')
        qc['qc_smooth_gaze_rate'] = float(sx.notna().mean()) if len(sx) else 0.0
    else:
        qc['qc_smooth_gaze_rate'] = None

    # Simple flagging rules (tune as needed)
    flag = 'ok'
    pdr = qc.get('qc_pupil_detection_rate')
    if isinstance(pdr, float) and pdr < 0.6:
        reasons.append('poor_pupil_detection_rate')
    bf = qc.get('qc_blink_fraction')
    if isinstance(bf, float) and bf > 0.35:
        reasons.append('excessive_blink_fraction')
    if duration > 0 and duration < 3.0:
        reasons.append('too_short_duration')

    if reasons:
        flag = 'review' if 'too_short_duration' not in reasons else 'invalid'

    return qc, flag, reasons

def angle_path_length(df):
    cols = ['yaw', 'pitch', 'roll']
    if not all(c in df.columns for c in cols):
        return 0.0, np.array([])
    a = df[cols].to_numpy()
    diffs = np.linalg.norm(np.diff(a, axis=0), axis=1)
    return float(np.nansum(diffs)), diffs

def angular_speed_stats(diffs, times):
    if len(diffs) == 0:
        return {'mean_speed': 0.0, 'speeds': np.array([])}
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
    speed_stats = angular_speed_stats(diffs, times)

    speeds = speed_stats['speeds']
    spikes = int(np.sum(speeds > spike_threshold)) if speeds.size else 0
    percent_time_moving = float(np.sum(speeds > motion_threshold) / (len(speeds)) ) if speeds.size else 0.0

    blink_rate = float(df['is_blinking'].sum()) / (duration+1e-6) if 'is_blinking' in df else 0.0
    
    # Gaze Dispersion logic will be handled in compute_saccade_metrics or here
    # We'll compute raw dispersion here for now
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


def _select_gaze_series(df, label="Unknown", vel_thresh=0.8):
    """
    Selects the best gaze signal (recorded smooth/est pixels, calibrated pixels, or raw ratio).
    Returns (times, signal, adjusted_vel_thresh, is_calibrated)
    """
    times = pd.to_numeric(df['t'], errors='coerce').to_numpy(dtype=float)

    # 0) Prefer task-runner recorded (already calibrated + filtered) if present
    for col in ('smooth_gaze_x', 'est_gaze_x'):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            if series.notna().sum() > 0:
                signal = series.to_numpy(dtype=float)
                is_calibrated = True
                if vel_thresh < 10.0:
                    adjusted_thresh = 200.0
                else:
                    adjusted_thresh = vel_thresh
                return times, signal, adjusted_thresh, is_calibrated

    # 1) Try Calibration (recompute from ratios if model exists)
    model = load_calibration_model(label)
    is_calibrated = apply_calibration(df, model)

    if is_calibrated:
        print(f"Applied calibration model for {label}.")
        signal = df['gaze_x_px'].to_numpy()

        if vel_thresh < 10.0:
            adjusted_thresh = 200.0
            print(f"  -> Auto-scaling velocity threshold to {adjusted_thresh} px/s")
        else:
            adjusted_thresh = vel_thresh
    else:
        print(f"No calibration found for {label}. Using raw ratios.")
        candidate_cols = ['g_horizontal', 'left_px', 'right_px']
        signal = None
        for col in candidate_cols:
            if col in df.columns:
                col_values = pd.to_numeric(df[col], errors='coerce')
                if col_values.notna().sum() > 0:
                    signal = col_values.to_numpy(dtype=float)
                    break
        if signal is None:
            signal = np.zeros_like(times)
        adjusted_thresh = vel_thresh

    return times, signal, adjusted_thresh, is_calibrated


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
    
    # Extract label from filename for calibration lookup
    filename = os.path.basename(csv_path)
    parts = filename.split('_')
    label = parts[0] if parts else "Unknown"

    times, pos, adjusted_thresh, is_calibrated = _select_gaze_series(df, label, vel_thresh)
    
    saccades = detect_saccades(times, pos, vel_thresh=adjusted_thresh, min_dur=min_dur, smooth_w=smooth_w)
    fixations = detect_fixations(times, pos, saccades, min_fix_dur=min_fix_dur)

    latencies = []
    if stimuli_times:
        latencies = saccade_latency_to_stimuli(saccades, stimuli_times, max_latency=latency_window)
        latencies = [lat if math.isfinite(lat) else None for lat in latencies]

    intrusive_total, intrusive_breakdown = (0, [])
    if interval_windows:
        intrusive_total, intrusive_breakdown = count_intrusive_saccades(saccades, interval_windows)

    # Re-calculate dispersion if calibrated to get pixel-based dispersion
    gaze_dispersion = float(np.std(pos)) if len(pos) > 0 else 0.0

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
        'is_calibrated': is_calibrated,
        'gaze_dispersion': gaze_dispersion # Overwrite with calibrated version if available
    }
    return metrics


def extract_stimuli_from_csv(df):
    """
    Extracts stimulus onset times from the 'stimulus_state' column.
    Returns a list of timestamps where state transitions to 'target'.
    """
    if 'stimulus_state' not in df.columns:
        return []
    
    # Create a mask for 'target' state
    is_target = df['stimulus_state'] == 'target'
    
    # Find transitions: current is target, previous was not target
    # shift(1) gives previous value. fillna(False) handles the first row.
    transitions = is_target & (~is_target.shift(1).fillna(False))
    
    # Get times corresponding to these transitions
    times = df.loc[transitions, 't'].tolist()
    return times


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
    raw_df = _load_csv_raw(csv_path)
    df = load_csv(csv_path)

    # Fallback: Extract stimuli times from CSV if not provided
    if not stimuli_times:
        stimuli_times = extract_stimuli_from_csv(df)
        if stimuli_times:
            print(f"Extracted {len(stimuli_times)} stimuli onsets from CSV.")

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

    # Task-specific: prioritize timing + direction (based on g_horizontal) when stimulus geometry is available.
    try:
        reaction_dir_metrics = compute_reaction_direction_metrics(
            csv_path=str(csv_path),
            df=df,
            vel_thresh=vel_thresh,
            min_dur=min_dur,
            smooth_w=smooth_w,
            latency_window=latency_window,
        )
        combined.update(reaction_dir_metrics)
    except Exception as e:  # pragma: no cover - analytics should not break summary
        combined['reaction_direction_error'] = str(e)

    qc, qc_flag, qc_reasons = _qc_metrics_and_flag(raw_df)
    combined.update(qc)
    combined['qc_flag'] = qc_flag
    combined['qc_reasons'] = qc_reasons

    return _clean_numeric(combined)


def _load_scalar_list(path):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        
        # FIX: Handle session summary dicts (e.g. {"stimuli_directions": [...]})
        if isinstance(data, dict):
            # Look for the list inside known keys
            found_list = None
            for key in ['stimuli_directions', 'stimuli_times', 'stimuli']:
                if key in data and isinstance(data[key], list):
                    found_list = data[key]
                    break
            
            if found_list is not None:
                data = found_list
            else:
                # If it's a dict but has no stimuli list, return empty
                return []

        # Handle list of dicts (e.g. stimuli_directions) or list of floats
        if isinstance(data, list):
            values = []
            for item in data:
                if isinstance(item, dict):
                    # Try to find a time-like key
                    # 'time' is used in stimuli_directions
                    val = item.get('time', item.get('timestamp', item.get('onset', 0)))
                    values.append(float(val))
                else:
                    values.append(float(item))
            return values
            
    except json.JSONDecodeError:
        pass
    
    # Fallback: Line-based parsing (CSV-like)
    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split(',')[0].strip()
        try:
            values.append(float(token))
        except ValueError:
            # Skip lines that aren't numbers (like JSON braces)
            continue
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
    parser.add_argument('--latency-window', type=float, default=2.0, help='Maximum latency window (s) when pairing stimuli to saccades.')
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