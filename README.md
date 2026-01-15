# ADHD Saccade Screening Toolkit

This fork repurposes the webcam eye-tracking stack to run short “games” (calibration, fixation, pro/antisaccade) and extract saccade metrics that may correlate with ADHD risk. The scripts show timed stimuli, log per-frame gaze/head data to CSV, summarize to JSON, and append feature rows to `master_metrics.csv` for later analysis.

## What you need
- Python 3.9+ with webcam access
- OpenCV, NumPy, Pandas, Dlib (install via `requirements.txt`) or use the provided `environment.yml`
- A laptop with a built-in or USB camera and a clear view of the participant’s eyes

## Setup
```bash
git clone https://github.com/antoinelame/GazeTracking.git
cd ADHD_Webcam

# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create --file environment.yml
conda activate GazeTracking
```

## Run order (recommended)
1) **Calibration** – fit gaze-to-screen mapping
```bash
python run_calibration.py --label P01
```
Saves model to `sessions/calibration/{label}_calibration.json` and offers a visual check.

2) **Fixation task** – sustained attention baseline
```bash
python run_fixation_task.py --label P01 --duration 25
```
Shows a center dot (optional), records `sessions/raw/*.csv`, writes summary JSON (if enabled), and appends metrics to `master_metrics.csv` with `task=fixation_dot`.

3) **Prosaccade task** – look TOWARD the flashed target
```bash
python run_prosaccade_task.py --label P01 --trials 24 --center-duration 1.0 --gap-duration 0.2 --target-duration 1.5
```

4) **Antisaccade task** – look OPPOSITE the flashed target
```bash
python run_antisaccade_task.py --label P01 --trials 24 --center-duration 1.0 --gap-duration 0.2 --target-duration 1.5
```

Press `q` in any window to exit early. Change `--trials` or timing flags to match your paradigm.

## Data outputs
- **Raw per-frame CSVs**: `sessions/raw/{label}_*.csv` (gaze ratios, head pose, blink flags, timestamps). With calibration enabled (i.e., after running calibration and supplying `--label`), tasks write additional calibrated columns:
  - Common columns: `t`, `yaw`, `pitch`, `roll`, `g_horizontal`, `is_blinking`, `left_px`, `left_py`, `right_px`, `right_py`
  - Added across tasks: `g_vertical`, `est_gaze_x`, `est_gaze_y`, `smooth_gaze_x`, `smooth_gaze_y`
  - Task stimuli (pro/anti): `trial_index`, `stimulus_state`, `stimulus_angle_deg`, `stimulus_x`, `stimulus_y`, `fixation_x`, `fixation_y`
- **Summaries**: `sessions/summaries/{label}_*.json` with latency distributions and task metadata.
- **Master metrics table**: `master_metrics.csv` appends one row per run. Key columns:
  - `session_label`, `task`, `raw_csv`, `configured_duration_s`, `configured_trials`
  - Timing: `center_duration_s`, `gap_duration_s`, `target_duration_s`, `duration_s`
  - Kinematics: `path_length_deg`, `mean_angular_speed_deg_per_s`, `spike_count`, `percent_time_moving`, `blink_rate_per_s`, `gaze_dispersion`
  - Saccade/fixation: `saccade_count`, `fixation_count`, `mean_saccade_duration_s`, `mean_saccade_peak_velocity`, `mean_saccade_amplitude`, `mean_fixation_duration_s`
  - Task-specific: `saccade_latencies_s` (list), `intrusive_saccade_count`, `intrusive_counts_per_interval`, `stimuli_directions`

## Calibration mapping (pixels)
- Run calibration once per participant or after setup changes:
  ```bash
  python run_calibration.py --label P01
  ```
- The tasks automatically look for `sessions/calibration/{label}_calibration.json` when `--label` is provided and, if found, map gaze ratios to pixel estimates:
  - `est_gaze_x = x_slope * g_horizontal + x_intercept`
  - `est_gaze_y = y_slope * g_vertical + y_intercept`
- A short moving average (window=8) produces `smooth_gaze_x/smooth_gaze_y` to reduce jitter.
- If no calibration file is found, tasks still run and write ratio-based columns; calibrated columns will be `NaN`.

## How the metrics are computed
- `analysis.py` loads each raw CSV, applies calibration if available, computes head-motion path length, angular speeds, blink rate, and gaze dispersion.
- Saccades/fixations are detected from gaze velocity; latencies are aligned to stimulus times; intrusive saccades are counted inside specified intervals.
- Outputs are cleaned (NaN → null) before writing JSON and `master_metrics.csv`.

## Tips for data quality
- Seat the participant ~50–70 cm from the screen; keep head/torso squared to the display.
- Put the webcam at eye height, centered above the screen; avoid looking up/down at the camera.
- Use even, front-facing light; avoid strong backlight or reflections on glasses.
- Keep the screen at a comfortable brightness and avoid moving windows behind stimuli.
- Minimize movement: steady chair/table; consider a chin rest or forehead stop if available.
- Re-run calibration after any camera move, lighting change, or participant repositioning.

## Quick demo (pupil tracking only)
If you just want to see the tracker overlay:
```bash
python example.py
```

## License
MIT License (original project by Antoine Lamé).
