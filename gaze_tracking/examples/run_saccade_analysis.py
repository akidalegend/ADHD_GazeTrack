import csv
import numpy as np
from gaze_tracking.saccades import detect_saccades, detect_fixations, saccade_latency_to_stimuli, count_intrusive_saccades
import sys

def load_session_csv(fn):
    t, g = [], []
    with open(fn, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t.append(float(row['t']))
            except:
                t.append(float('nan'))
            gval = row.get('g_horizontal')
            if gval is None or gval == '':
                gval = row.get('left_px', '')
            try:
                g.append(float(gval))
            except:
                g.append(float('nan'))
    return np.array(t), np.array(g)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python examples/run_saccade_analysis.py path/to/session.csv")
        sys.exit(1)
    fn = sys.argv[1]
    times, pos = load_session_csv(fn)
    saccades = detect_saccades(times, pos, vel_thresh=0.8, min_dur=0.015, smooth_w=5)
    fixs = detect_fixations(times, pos, saccades, min_fix_dur=0.08)
    print(f"Detected {len(saccades)} saccades, {len(fixs)} fixations")
    # example latency / intrusive counts (replace with real stimuli/intervals)
    stimuli = [times[0] + 1.0] if len(times) else []
    lat = saccade_latency_to_stimuli(saccades, stimuli, max_latency=1.0)
    print("Latencies (s):", lat)
    intervals = [(times[0]+2.0, times[0]+3.0)] if len(times) else []
    total_intrusive, per_interval = count_intrusive_saccades(saccades, intervals)
    print("Intrusive:", total_intrusive, per_interval)