import argparse
import json
import math
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gaze_tracking import GazeTracking
from task_utils import ensure_directories, prompt_label


def _wait_for_click(
    window_name: str = "Calibration Start",
    width: int = 600,
    height: int = 400,
    text: str = "Click to start",
) -> bool:
    """Display a simple start screen and wait for a left-click or 'q'/ESC."""
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
        cv2.rectangle(
            canvas,
            (text_x - 20, text_y - text_size[1] - 20),
            (text_x + text_size[0] + 20, text_y + 20),
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF
        if clicked:
            cv2.destroyWindow(window_name)
            return True
        if key in (ord("q"), 27):  # q or ESC
            cv2.destroyWindow(window_name)
            return False


def _countdown(
    window_name: str,
    width: int,
    height: int,
    win_x: int,
    win_y: int,
    seconds: int = 3,
    message: str = "Starting calibration",
) -> bool:
    """Show a brief countdown before calibration begins."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, win_x, win_y)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for remaining in range(seconds, 0, -1):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        label_text = f"{message} in {remaining}"
        text_size = cv2.getTextSize(label_text, font, 1.2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(canvas, label_text, (text_x, text_y), font, 1.2, (0, 255, 0), 3)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(1000) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            cv2.destroyWindow(window_name)
            return False

    cv2.destroyWindow(window_name)
    return True


def get_screen_resolution(default_w: int = 1920, default_h: int = 1080) -> Tuple[int, int]:
    """Return a safe window size; caller can override via CLI."""
    return default_w, default_h


def draw_calibration_target(frame: np.ndarray, x: int, y: int, radius: int = 15, color=(0, 0, 255)) -> None:
    """Draw a target (dot with center)."""
    cv2.circle(frame, (x, y), radius, color, -1)
    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)


# --- One Euro filter (for calibrated gaze smoothing) ---
class _LowPass:
    def __init__(self) -> None:
        self._y: Optional[float] = None

    def reset(self) -> None:
        self._y = None

    def apply(self, x: float, a: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = a * x + (1.0 - a) * self._y
        return float(self._y)


class OneEuroFilter:
    def __init__(self, *, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._t_prev: Optional[float] = None
        self._x_prev: Optional[float] = None
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

        dx = (float(x) - float(self._x_prev)) / dt if dt > 0 else 0.0
        self._x_prev = float(x)

        a_d = self._alpha(self.d_cutoff, dt)
        edx = self._dx.apply(dx, a_d)

        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(cutoff, dt)
        return self._x.apply(float(x), a)


# --- 2D polynomial calibration helpers ---
def _poly2_features(h: float, v: float) -> np.ndarray:
    # [1, h, v, h*v, h^2, v^2]
    return np.array([1.0, float(h), float(v), float(h) * float(v), float(h) ** 2, float(v) ** 2], dtype=float)


def _poly2_design_matrix(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    ones = np.ones_like(h, dtype=float)
    return np.column_stack([ones, h, v, h * v, h * h, v * v]).astype(float)


def _fit_poly2(h: np.ndarray, v: np.ndarray, target: np.ndarray) -> List[float]:
    a = _poly2_design_matrix(h, v)
    coef, _, _, _ = np.linalg.lstsq(a, target, rcond=None)
    return [float(c) for c in coef.tolist()]


def _apply_model(model: dict, h: float, v: float) -> Tuple[float, float]:
    """Apply either poly2 model (preferred) or linear fallback."""
    if model and model.get("type") == "poly2" and "x_coef" in model and "y_coef" in model:
        feats = _poly2_features(h, v)
        x = float(np.dot(np.array(model["x_coef"], dtype=float), feats))
        y = float(np.dot(np.array(model["y_coef"], dtype=float), feats))
        return x, y

    # linear fallback
    x = float(model.get("x_slope", 1.0)) * float(h) + float(model.get("x_intercept", 0.0))
    y = float(model.get("y_slope", 1.0)) * float(v) + float(model.get("y_intercept", 0.0))
    return x, y


def collect_calibration_points(
    session_label: str,
    calib_w: int,
    calib_h: int,
    win_x: int,
    win_y: int,
    fullscreen: bool = False,
) -> Optional[dict]:
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Unable to access camera.")
        return None

    # Proposed change: push targets closer to true edges/corners.
    points_norm = [
        (0.05, 0.05), (0.50, 0.05), (0.95, 0.05),
        (0.05, 0.50), (0.50, 0.50), (0.95, 0.50),
        (0.05, 0.95), (0.50, 0.95), (0.95, 0.95),
    ]

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, calib_w, calib_h)
    cv2.moveWindow(window_name, win_x, win_y)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calibration_data = {
        "label": session_label,
        "timestamp": time.time(),
        "screen_width": calib_w,
        "screen_height": calib_h,
        "points": [],
    }

    print(f"Starting calibration for {session_label}. Look at the red dots.")

    for i, (px, py) in enumerate(points_norm):
        target_x = int(px * calib_w)
        target_y = int(py * calib_h)

        # 3.5 seconds total (1.5 settle, 2.0 record)
        start_time = time.time()
        samples_h: List[float] = []
        samples_v: List[float] = []
        samples_total = 0
        samples_blink = 0

        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Error: Failed to read from camera.")
                webcam.release()
                cv2.destroyAllWindows()
                return None

            stimulus = np.zeros((calib_h, calib_w, 3), dtype=np.uint8)
            draw_calibration_target(stimulus, target_x, target_y)
            cv2.putText(
                stimulus,
                f"Point {i + 1}/9",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            gaze.refresh(frame)
            elapsed = time.time() - start_time

            if elapsed > 1.5:
                samples_total += 1
                if gaze.pupils_located and not gaze.is_blinking():
                    h_ratio = gaze.horizontal_ratio()
                    v_ratio = gaze.vertical_ratio()
                    if h_ratio is not None and v_ratio is not None:
                        samples_h.append(float(h_ratio))
                        samples_v.append(float(v_ratio))
                else:
                    if gaze.is_blinking():
                        samples_blink += 1

            cv2.imshow(window_name, stimulus)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                webcam.release()
                cv2.destroyAllWindows()
                return None

            if elapsed > 3.5:
                break

        if samples_h and samples_v:
            avg_h = float(np.median(samples_h))
            avg_v = float(np.median(samples_v))
            calibration_data["points"].append(
                {
                    "target_x": int(target_x),
                    "target_y": int(target_y),
                    "gaze_h": avg_h,
                    "gaze_v": avg_v,
                    "samples": int(len(samples_h)),
                    "samples_total": int(samples_total),
                    "samples_blink": int(samples_blink),
                }
            )
        else:
            print(f"Warning: No valid gaze data for point {i + 1}")
            calibration_data["points"].append(
                {
                    "target_x": int(target_x),
                    "target_y": int(target_y),
                    "gaze_h": None,
                    "gaze_v": None,
                    "samples": 0,
                    "samples_total": int(samples_total),
                    "samples_blink": int(samples_blink),
                }
            )

    webcam.release()
    cv2.destroyAllWindows()
    return calibration_data


def compute_calibration_model(data: dict) -> Optional[dict]:
    """
    Fit a 2D (degree-2) polynomial mapping from (gaze_h, gaze_v) -> (screen_x, screen_y).
    Also saves a linear fallback (x_slope/x_intercept, y_slope/y_intercept) for compatibility.
    """
    points = data.get("points", [])

    valid = [p for p in points if p.get("gaze_h") is not None and p.get("gaze_v") is not None]
    if len(valid) < 4:
        print("Not enough points for calibration.")
        return None

    tx = np.array([p["target_x"] for p in valid], dtype=float)
    ty = np.array([p["target_y"] for p in valid], dtype=float)
    gh = np.array([p["gaze_h"] for p in valid], dtype=float)
    gv = np.array([p["gaze_v"] for p in valid], dtype=float)

    # Always compute linear fallback (for other scripts that expect these keys)
    poly_x = np.polyfit(gh, tx, 1)
    poly_y = np.polyfit(gv, ty, 1)
    model = {
        "x_slope": float(poly_x[0]),
        "x_intercept": float(poly_x[1]),
        "y_slope": float(poly_y[0]),
        "y_intercept": float(poly_y[1]),
        "type": "linear",
    }

    # Preferred: poly2 if enough points (needs at least 6 for full rank in practice)
    if len(valid) >= 6:
        try:
            x_coef = _fit_poly2(gh, gv, tx)
            y_coef = _fit_poly2(gh, gv, ty)
            model.update(
                {
                    "type": "poly2",
                    "x_coef": x_coef,  # [1, h, v, h*v, h^2, v^2]
                    "y_coef": y_coef,
                }
            )
        except Exception as e:
            print(f"Warning: poly2 fit failed, falling back to linear: {e}")

    return model


def _compute_fit_metrics(points: Sequence[dict], model: Optional[dict]) -> Optional[dict]:
    """Compute in-sample pixel error metrics on valid calibration points using the saved model."""
    valid = [p for p in points if p.get("gaze_h") is not None and p.get("gaze_v") is not None]
    if not valid or not model:
        return None

    preds = []
    targets = []
    for p in valid:
        x, y = _apply_model(model, float(p["gaze_h"]), float(p["gaze_v"]))
        preds.append((x, y))
        targets.append((float(p["target_x"]), float(p["target_y"])))

    preds_arr = np.array(preds, dtype=float)
    targets_arr = np.array(targets, dtype=float)
    d = preds_arr - targets_arr
    dist = np.sqrt(np.sum(d * d, axis=1))

    return {
        "model_type": str(model.get("type", "linear")),
        "n_points_valid": int(len(valid)),
        "rmse_px": float(np.sqrt(np.mean(dist * dist))) if dist.size else None,
        "mae_px": float(np.mean(np.abs(dist))) if dist.size else None,
        "p95_px": float(np.percentile(dist, 95)) if dist.size else None,
    }


def verify_calibration(model: dict, screen_w: int, screen_h: int, win_x: int, win_y: int, fullscreen: bool = False) -> None:
    """Verification loop: show estimated gaze point. Uses One-Euro smoothing."""
    print("\n--- VERIFICATION MODE ---")
    print("Look around the screen. A green circle should follow your eyes.")
    print("Press 'q' or ESC to finish.")

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Unable to access camera.")
        return

    window_name = "Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_w, screen_h)
    cv2.moveWindow(window_name, win_x, win_y)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Proposed change: One-Euro smoothing instead of moving average
    x_filt = OneEuroFilter(min_cutoff=0.5, beta=0.005, d_cutoff=1.0)
    y_filt = OneEuroFilter(min_cutoff=0.5, beta=0.005, d_cutoff=1.0)

    max_x = max(0, int(screen_w) - 1)
    max_y = max(0, int(screen_h) - 1)

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gaze.refresh(frame)

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "Verification: Look around.",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "Green Dot = Estimated Gaze (One-Euro smoothed)",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if gaze.pupils_located:
            if gaze.is_blinking():
                # avoid filter “catch-up” jumps after blinks
                x_filt.reset()
                y_filt.reset()
            else:
                h_ratio = gaze.horizontal_ratio()
                v_ratio = gaze.vertical_ratio()
                if h_ratio is not None and v_ratio is not None:
                    now = time.time()

                    raw_x_f, raw_y_f = _apply_model(model, float(h_ratio), float(v_ratio))
                    smooth_x_f = x_filt.update(now, raw_x_f)
                    smooth_y_f = y_filt.update(now, raw_y_f)

                    smooth_x = int(round(smooth_x_f))
                    smooth_y = int(round(smooth_y_f))

                    smooth_x = max(0, min(max_x, smooth_x))
                    smooth_y = max(0, min(max_y, smooth_y))

                    cv2.circle(canvas, (smooth_x, smooth_y), 20, (0, 255, 0), -1)

                    annotated = gaze.annotated_frame()
                    small_frame = cv2.resize(annotated, (320, 240))
                    canvas[screen_h - 240 : screen_h, 0:320] = small_frame

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="Participant label")
    parser.add_argument("--stimulus-width", type=int, default=1400, help="Stimulus window width in pixels")
    parser.add_argument("--stimulus-height", type=int, default=900, help="Stimulus window height in pixels")
    parser.add_argument("--window-x", type=int, default=0, help="Top-left X position for stimulus window (use to choose monitor)")
    parser.add_argument("--window-y", type=int, default=0, help="Top-left Y position for stimulus window")
    parser.add_argument("--fullscreen", action="store_true", help="Force stimulus window to fill chosen monitor (overrides width/height)")
    args = parser.parse_args()

    label = args.label if args.label else prompt_label()

    ensure_directories(["sessions/calibration"])

    print("Loading gaze tracking model...")
    _ = GazeTracking()  # Preload MediaPipe tracker
    print("Gaze tracking model loaded successfully!")

    print("\nCalibration instructions:")
    print("- Sit ~60cm from the screen with steady lighting.")
    print("- A 'Calibration Start' window appears; click to begin, or press q/ESC to cancel.")
    print("- Look at each red dot; press q or Esc at any time to abort.")
    print("- After dots finish, a verification view runs; press q/Esc to exit when done.\n")

    if not _wait_for_click():
        print("Calibration cancelled before start click.")
        raise SystemExit(0)

    screen_w, screen_h = get_screen_resolution(args.stimulus_width, args.stimulus_height)

    if not _countdown(
        window_name="Calibration Countdown",
        width=screen_w,
        height=screen_h,
        win_x=args.window_x,
        win_y=args.window_y,
        seconds=3,
        message="Calibration starts",
    ):
        print("Calibration cancelled during countdown.")
        raise SystemExit(0)

    cal_data = collect_calibration_points(label, screen_w, screen_h, args.window_x, args.window_y, args.fullscreen)

    if not cal_data:
        print("Calibration aborted.")
        raise SystemExit(0)

    model = compute_calibration_model(cal_data)
    if not model:
        print("Calibration failed to generate a model.")
        raise SystemExit(1)

    cal_data["model"] = model
    fit_metrics = _compute_fit_metrics(cal_data.get("points", []), model)
    cal_data["fit_metrics"] = fit_metrics

    print("\nCalibration Successful!")
    print(f"Model type: {model.get('type', 'linear')}")
    print(f"Linear fallback X: ScreenX = {model['x_slope']:.2f} * GazeH + {model['x_intercept']:.2f}")
    print(f"Linear fallback Y: ScreenY = {model['y_slope']:.2f} * GazeV + {model['y_intercept']:.2f}")
    if model.get("type") == "poly2":
        print("Poly2 coefficients saved (x_coef/y_coef).")
    if fit_metrics:
        print(
            "Fit error (in-sample): "
            f"RMSE={fit_metrics['rmse_px']:.1f}px, "
            f"MAE={fit_metrics['mae_px']:.1f}px, "
            f"P95={fit_metrics['p95_px']:.1f}px "
            f"(n={fit_metrics['n_points_valid']}, model={fit_metrics.get('model_type')})"
        )

    filename = f"sessions/calibration/{label}_calibration.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cal_data, f, indent=4)
    print(f"Saved to {filename}")

    verify_calibration(model, screen_w, screen_h, args.window_x, args.window_y, args.fullscreen)