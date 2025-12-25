import cv2
import numpy as np
import time
import json
import argparse
from gaze_tracking import GazeTracking
from task_utils import prompt_label, ensure_directories


def _wait_for_click(window_name: str = "Calibration Start", width: int = 600, height: int = 400, text: str = "Click to start") -> bool:
    """Display a simple start screen and wait for a left-click or 'q'."""
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
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return False


def _countdown(window_name: str, width: int, height: int, win_x: int, win_y: int, seconds: int = 3, message: str = "Starting calibration") -> bool:
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
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return False

    cv2.destroyWindow(window_name)
    return True

def get_screen_resolution(default_w=1920, default_h=1080):
    """Return a safe window size; caller can override via CLI."""
    return default_w, default_h

def draw_calibration_target(frame, x, y, radius=15, color=(0, 0, 255)):
    """Draws a target (dot with center) on the frame."""
    cv2.circle(frame, (x, y), radius, color, -1)
    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

def collect_calibration_points(session_label, calib_w, calib_h, win_x, win_y, fullscreen=False):
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    # Get screen dimensions for scaling the window
    # use provided window size so calibration stays on the chosen display
    
    # Define 9 calibration points (normalized 0.0 to 1.0)
    # Top-Left, Top-Mid, Top-Right, Mid-Left, Center, Mid-Right, Bot-Left, Bot-Mid, Bot-Right
    points_norm = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
    ]
    
    # Create a full-screen window
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
        "points": []
    }

    print(f"Starting calibration for {session_label}. Look at the red dots.")
    
    for i, (px, py) in enumerate(points_norm):
        # Convert normalized points to screen coordinates
        target_x = int(px * calib_w)
        target_y = int(py * calib_h)
        
        # Show point for 3.5 seconds (1.5s settle, 2.0s record)
        start_time = time.time()
        samples_x = []
        samples_y = []
        
        while True:
            _, frame = webcam.read()
            
            # Create a black background image for the stimulus
            stimulus = np.zeros((calib_h, calib_w, 3), dtype=np.uint8)
            
            # Draw the target
            draw_calibration_target(stimulus, target_x, target_y)
            
            # Add instruction text
            cv2.putText(stimulus, f"Point {i+1}/9", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Process gaze
            gaze.refresh(frame)
            elapsed = time.time() - start_time
            
            # Record data after settle window to reduce jitter
            if elapsed > 1.5:
                if gaze.pupils_located:
                    # We use the raw pupil coordinates or ratios
                    # Here we collect horizontal/vertical ratios if available, 
                    # or raw pupil coords relative to eye frame
                    
                    # Note: GazeTracking library mainly exposes horizontal_ratio and vertical_ratio
                    h_ratio = gaze.horizontal_ratio()
                    v_ratio = gaze.vertical_ratio()
                    
                    if h_ratio is not None and v_ratio is not None:
                        samples_x.append(h_ratio)
                        samples_y.append(v_ratio)
            
            # Show the stimulus window
            cv2.imshow(window_name, stimulus)
            
            if cv2.waitKey(1) == 27: # ESC
                webcam.release()
                cv2.destroyAllWindows()
                return None

            if elapsed > 3.5: # Move to next point after extended window
                break
        
        # Average the samples for this point
        if samples_x and samples_y:
            # Use Median instead of Mean to filter out blinks/outliers
            avg_h = np.median(samples_x)
            avg_v = np.median(samples_y)
            calibration_data["points"].append({
                "target_x": target_x,
                "target_y": target_y,
                "gaze_h": float(avg_h),
                "gaze_v": float(avg_v),
                "samples": len(samples_x)
            })
        else:
            print(f"Warning: No valid gaze data for point {i+1}")

    webcam.release()
    cv2.destroyAllWindows()
    return calibration_data

def compute_calibration_model(data):
    """
    Simple linear regression to map Gaze Ratios -> Screen Pixels.
    x_screen = a * gaze_h + b
    y_screen = c * gaze_v + d
    """
    points = data["points"]
    if len(points) < 4:
        print("Not enough points for calibration.")
        return None

    # Extract arrays
    targets_x = np.array([p["target_x"] for p in points])
    targets_y = np.array([p["target_y"] for p in points])
    gaze_h = np.array([p["gaze_h"] for p in points])
    gaze_v = np.array([p["gaze_v"] for p in points])

    # Fit linear polynomial (degree 1)
    # Returns [slope, intercept]
    poly_x = np.polyfit(gaze_h, targets_x, 1) 
    poly_y = np.polyfit(gaze_v, targets_y, 1)

    model = {
        "x_slope": float(poly_x[0]),
        "x_intercept": float(poly_x[1]),
        "y_slope": float(poly_y[0]),
        "y_intercept": float(poly_y[1])
    }
    return model

def verify_calibration(model, screen_w, screen_h, win_x, win_y, fullscreen=False):
    """
    Runs a loop showing the estimated gaze point on screen.
    Allows the user to visually check accuracy.
    """
    print("\n--- VERIFICATION MODE ---")
    print("Look around the screen. A green circle should follow your eyes.")
    print("Press 'q' or ESC to finish.")
    
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    window_name = "Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_w, screen_h)
    cv2.moveWindow(window_name, win_x, win_y)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        _, frame = webcam.read()
        gaze.refresh(frame)
        
        # Create a black background
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        # Instructions
        cv2.putText(canvas, "Verification: Look around.", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Green Dot = Estimated Gaze", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if gaze.pupils_located:
            h_ratio = gaze.horizontal_ratio()
            v_ratio = gaze.vertical_ratio()
            
            if h_ratio is not None and v_ratio is not None:
                # Apply the model
                # x = slope * h + intercept
                est_x = int(model["x_slope"] * h_ratio + model["x_intercept"])
                est_y = int(model["y_slope"] * v_ratio + model["y_intercept"])
                
                # Clamp to screen
                est_x = max(0, min(screen_w, est_x))
                est_y = max(0, min(screen_h, est_y))
                
                # Draw cursor
                cv2.circle(canvas, (est_x, est_y), 20, (0, 255, 0), -1)
                
                # Draw raw eyes on corner for debug
                annotated = gaze.annotated_frame()
                small_frame = cv2.resize(annotated, (320, 240))
                canvas[screen_h-240:screen_h, 0:320] = small_frame

        cv2.imshow(window_name, canvas)
        
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or q
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
    gaze_model = GazeTracking()  # Preload dlib model here
    print("Gaze tracking model loaded successfully!")
    
    print("Ensure the user is sitting 60cm from the screen.")
    print("Ensure lighting is consistent.")
    print("Click the start window to begin calibration or press q to cancel.")

    if not _wait_for_click():
        print("Calibration cancelled before start click.")
        exit(0)

    screen_w, screen_h = get_screen_resolution(args.stimulus_width, args.stimulus_height)

    if not _countdown(
        window_name="Calibration Countdown",
        width=screen_w,
        height=screen_h,
        win_x=args.window_x,
        win_y=args.window_y,
        seconds=3,
        message="Calibration starts"
    ):
        print("Calibration cancelled during countdown.")
        exit(0)

    cal_data = collect_calibration_points(label, screen_w, screen_h, args.window_x, args.window_y, args.fullscreen)
    
    if cal_data:
        model = compute_calibration_model(cal_data)
        if model:
            cal_data["model"] = model
            print("\nCalibration Successful!")
            print(f"X Model: ScreenX = {model['x_slope']:.2f} * GazeH + {model['x_intercept']:.2f}")
            print(f"Y Model: ScreenY = {model['y_slope']:.2f} * GazeV + {model['y_intercept']:.2f}")
            
            filename = f"sessions/calibration/{label}_calibration.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(cal_data, f, indent=4)
            print(f"Saved to {filename}")
            
            # Verify calibration
            verify_calibration(model, screen_w, screen_h, args.window_x, args.window_y, args.fullscreen)
        else:
            print("Calibration failed to generate a model.")
    else:
        print("Calibration aborted.")