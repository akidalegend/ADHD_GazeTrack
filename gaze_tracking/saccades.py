import numpy as np

def _moving_average(x, w=5):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')

def detect_saccades(times, pos, vel_thresh=0.5, min_dur=0.02, smooth_w=5):
    times = np.asarray(times, dtype=float)
    pos = np.asarray(pos, dtype=float)

    mask = np.isfinite(pos) & np.isfinite(times)
    if mask.sum() < 3:
        return []

    pos_interp = np.copy(pos)
    if not np.all(mask):
        pos_interp[~mask] = np.interp(times[~mask], times[mask], pos[mask])

    pos_s = _moving_average(pos_interp, w=smooth_w)

    dt = np.diff(times, prepend=times[0])
    dt[dt == 0] = 1e-6
    vel = np.abs(np.gradient(pos_s, times))

    active = vel > vel_thresh
    saccades = []
    i = 0
    N = len(active)
    while i < N:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < N and active[i]:
            i += 1
        end = i - 1
        duration = times[end] - times[start]
        if duration >= min_dur:
            amp = pos_s[end] - pos_s[start]
            peak = float(np.max(vel[start:end+1]))
            saccades.append({
                'onset_idx': int(start),
                'offset_idx': int(end),
                'onset_t': float(times[start]),
                'offset_t': float(times[end]),
                'duration': float(duration),
                'peak_velocity': peak,
                'amplitude': float(amp)
            })
    return saccades

def detect_fixations(times, pos, saccades, min_fix_dur=0.08):
    times = np.asarray(times, dtype=float)
    pos = np.asarray(pos, dtype=float)
    N = len(times)

    sac_mask = np.zeros(N, dtype=bool)
    for s in saccades:
        sac_mask[s['onset_idx']:s['offset_idx']+1] = True

    fixations = []
    i = 0
    while i < N:
        if sac_mask[i]:
            i += 1
            continue
        start = i
        while i < N and not sac_mask[i]:
            i += 1
        end = i - 1
        duration = times[end] - times[start]
        if duration >= min_fix_dur:
            fixations.append({
                'start_idx': int(start),
                'end_idx': int(end),
                'start_t': float(times[start]),
                'end_t': float(times[end]),
                'duration': float(duration),
                'pos_mean': float(np.nanmean(pos[start:end+1]))
            })
    return fixations

def saccade_latency_to_stimuli(saccades, stimuli_times, max_latency=1.0):
    onsets = np.array([s['onset_t'] for s in saccades]) if saccades else np.array([])
    latencies = []
    for stim in stimuli_times:
        if onsets.size == 0:
            latencies.append(float('nan'))
            continue
        idx = np.searchsorted(onsets, stim, side='left')
        if idx < len(onsets) and onsets[idx] - stim <= max_latency:
            latencies.append(float(onsets[idx] - stim))
        else:
            latencies.append(float('nan'))
    return latencies

def count_intrusive_saccades(saccades, intervals):
    onsets = np.array([s['onset_t'] for s in saccades]) if saccades else np.array([])
    counts = []
    for (a,b) in intervals:
        if onsets.size == 0:
            counts.append(0)
        else:
            counts.append(int(np.sum((onsets >= a) & (onsets <= b))))
    return int(np.sum(counts)), counts