"""
MediaPipe-based gaze tracking implementation.
Provides similar interface to GazeTracking but using MediaPipe Face Mesh + Iris detection.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple

try:
    import mediapipe as mp
    try:
        # Prefer top-level solutions when available
        from mediapipe import solutions as mp_solutions
    except Exception:
        # Fallback for environments where solutions is under mediapipe.python
        from mediapipe.python import solutions as mp_solutions
except ImportError:
    raise ImportError("mediapipe not installed. Run: pip install mediapipe")


class GazeTrackingMediaPipe:
    """
    Gaze tracking using MediaPipe Face Mesh and Iris detection.
    Provides compatible interface with dlib-based GazeTracking.
    """

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, float(x))))

    @staticmethod
    def _ratio_1d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Optional[float]:
        # Project p onto the axis a->b and normalize by |a-b|^2
        d = b - a
        denom = float(np.dot(d, d))
        if denom <= 1e-6:
            return None
        return float(np.dot(p - a, d) / denom)

    def __init__(self):
        self.frame = None
        self.frame_h = None
        self.frame_w = None
        
        # MediaPipe components (support both package layouts)
        self.mp_face_mesh = mp_solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State
        self.head_pose = None
        self.pupils_located = False
        self.left_iris = None  # (x, y) in frame coords
        self.right_iris = None  # (x, y) in frame coords
        self.left_eye_landmarks = None  # Mediapipe eye landmarks
        self.right_eye_landmarks = None
        
        # Blink detection: eye aspect ratio history
        self._blink_history = deque(maxlen=5)
        self._blinking = False

        # Stable eye-frame ratios (reduces fixation micro-jitter)
        self._h_hist = deque(maxlen=3)
        self._v_hist = deque(maxlen=3)

        # Cached landmarks used for stable ratios
        self._left_corner_l = None
        self._left_corner_r = None
        self._right_corner_l = None
        self._right_corner_r = None
        self._left_top = None
        self._left_bottom = None
        self._right_top = None
        self._right_bottom = None

    def refresh(self, frame: np.ndarray) -> None:
        """
        Process frame and extract gaze features.
        
        Args:
            frame: BGR image from camera
        """
        self.frame = frame
        self.frame_h, self.frame_w = frame.shape[:2]
        
        # Run face mesh
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(frame_rgb)
        
        if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
            landmarks = face_results.multi_face_landmarks[0]
            self._extract_pupils_and_pose(landmarks, frame)
        else:
            self.pupils_located = False
            self.head_pose = None
            self.left_iris = None
            self.right_iris = None
            self.left_eye_landmarks = None
            self.right_eye_landmarks = None
           
            self._h_hist.clear()
            self._v_hist.clear()

    def _extract_pupils_and_pose(self, landmarks, frame: np.ndarray) -> None:
        """Extract iris positions and head pose from face landmarks."""
        
        # MediaPipe landmark indices for eyes
        LEFT_EYE_INDICES = [33, 133, 145, 153, 154, 155, 156, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE_INDICES = [263, 362, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 466]
        
        # MediaPipe iris landmark indices (center of iris)
        LEFT_IRIS_CENTER = 468
        RIGHT_IRIS_CENTER = 473
        
        # Get eye regions
        left_eye_pts = np.array([
            [landmarks.landmark[i].x * self.frame_w, landmarks.landmark[i].y * self.frame_h]
            for i in LEFT_EYE_INDICES
        ], dtype=np.float32)
        
        right_eye_pts = np.array([
            [landmarks.landmark[i].x * self.frame_w, landmarks.landmark[i].y * self.frame_h]
            for i in RIGHT_EYE_INDICES
        ], dtype=np.float32)
        
        # Use MediaPipe's built-in iris landmarks (more accurate than contour detection)
        try:
            if len(landmarks.landmark) > RIGHT_IRIS_CENTER:
                left_iris_x = (landmarks.landmark[LEFT_IRIS_CENTER].x * self.frame_w)
                left_iris_y = (landmarks.landmark[LEFT_IRIS_CENTER].y * self.frame_h)
                right_iris_x = (landmarks.landmark[RIGHT_IRIS_CENTER].x * self.frame_w)
                right_iris_y = (landmarks.landmark[RIGHT_IRIS_CENTER].y * self.frame_h)
                
                self.left_iris = (left_iris_x, left_iris_y)
                self.right_iris = (right_iris_x, right_iris_y)
                self.pupils_located = True
            else:
                # Fallback to contour-based detection if iris landmarks not available
                left_iris_center = self._extract_iris_center(frame, left_eye_pts.astype(np.int32))
                right_iris_center = self._extract_iris_center(frame, right_eye_pts.astype(np.int32))
                
                if left_iris_center and right_iris_center:
                    self.left_iris = left_iris_center
                    self.right_iris = right_iris_center
                    self.pupils_located = True
                else:
                    self.pupils_located = False
        except (IndexError, AttributeError):
            self.pupils_located = False
        
        # Estimate head pose
        self.head_pose = self._estimate_head_pose(landmarks, frame)
        
        # Detect blink
        self._detect_blink(landmarks)

        def lm_xy(idx: int) -> np.ndarray:
            return np.array(
                [landmarks.landmark[idx].x * self.frame_w, landmarks.landmark[idx].y * self.frame_h],
                dtype=np.float32,
            )

        # Corners chosen so ratio increases to the RIGHT in image coordinates
        self._left_corner_l = lm_xy(33)    # left eye outer (left in image)
        self._left_corner_r = lm_xy(133)   # left eye inner (right in image)

        self._right_corner_l = lm_xy(362)  # right eye inner (left in image)
        self._right_corner_r = lm_xy(263)  # right eye outer (right in image)

        # Vertical reference (top->bottom)
        self._left_top = lm_xy(159)
        self._left_bottom = lm_xy(145)

        self._right_top = lm_xy(386)
        self._right_bottom = lm_xy(374)
        
        # Store eye landmarks for reference
        self.left_eye_landmarks = left_eye_pts
        self.right_eye_landmarks = right_eye_pts

    def _extract_iris_center(self, frame: np.ndarray, eye_pts: np.ndarray) -> Optional[Tuple[int, int]]:
        """Extract iris center from eye region using thresholding and contours."""
        if len(eye_pts) < 3:
            return None
        
        # Create mask for eye region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [eye_pts], 255)
        
        # Extract eye region
        h, w = frame.shape[:2]
        x_min, x_max = np.clip([eye_pts[:, 0].min(), eye_pts[:, 0].max()], 0, w)
        y_min, y_max = np.clip([eye_pts[:, 1].min(), eye_pts[:, 1].max()], 0, h)
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        if eye_region.size == 0:
            return None
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding to find dark pupil
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        
        if not contours:
            return None
        
        # Get largest contour (should be iris)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute centroid
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]) + x_min
            cy = int(M["m01"] / M["m00"]) + y_min
            return (cx, cy)
        
        return None

    def _estimate_head_pose(self, landmarks, frame: np.ndarray) -> Optional[dict]:
        """Estimate head pose (yaw, pitch, roll) from MediaPipe landmarks."""
        # Key landmark indices
        NOSE = 1
        CHIN = 152
        LEFT_EYE = 33
        RIGHT_EYE = 263
        LEFT_MOUTH = 287
        RIGHT_MOUTH = 57
        
        image_points = np.array([
            [landmarks.landmark[NOSE].x * self.frame_w, landmarks.landmark[NOSE].y * self.frame_h],
            [landmarks.landmark[CHIN].x * self.frame_w, landmarks.landmark[CHIN].y * self.frame_h],
            [landmarks.landmark[LEFT_EYE].x * self.frame_w, landmarks.landmark[LEFT_EYE].y * self.frame_h],
            [landmarks.landmark[RIGHT_EYE].x * self.frame_w, landmarks.landmark[RIGHT_EYE].y * self.frame_h],
            [landmarks.landmark[LEFT_MOUTH].x * self.frame_w, landmarks.landmark[LEFT_MOUTH].y * self.frame_h],
            [landmarks.landmark[RIGHT_MOUTH].x * self.frame_w, landmarks.landmark[RIGHT_MOUTH].y * self.frame_h],
        ], dtype=np.float32)
        
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float32)
        
        # Camera matrix
        focal_length = self.frame_w
        center = (self.frame_w / 2, self.frame_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # Convert to Euler angles
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        angles = {
            'pitch': math.degrees(x),
            'yaw': math.degrees(y),
            'roll': math.degrees(z)
        }
        
        # Project axis for visualization
        axis_3d = np.float32([[100.0, 0.0, 0.0],
                              [0.0, 100.0, 0.0],
                              [0.0, 0.0, 100.0]])
        nose_3d = np.float32([[0.0, 0.0, 0.0]])
        nose_point_2d, _ = cv2.projectPoints(nose_3d, rvec, tvec, camera_matrix, dist_coeffs)
        axis_points_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        
        nose_point = (int(nose_point_2d[0][0][0]), int(nose_point_2d[0][0][1]))
        axis_points = [(int(p[0][0]), int(p[0][1])) for p in axis_points_2d]
        
        return {
            'rvec': rvec,
            'tvec': tvec,
            'angles': angles,
            'nose_point': nose_point,
            'axis_points': axis_points,
            'image_points': image_points
        }

    def _detect_blink(self, landmarks) -> None:
        """Detect blink by computing eye aspect ratio."""
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        LEFT_EYE_LEFT = 33
        LEFT_EYE_RIGHT = 133
        
        eye_top = np.array([landmarks.landmark[LEFT_EYE_TOP].x * self.frame_w, landmarks.landmark[LEFT_EYE_TOP].y * self.frame_h])
        eye_bottom = np.array([landmarks.landmark[LEFT_EYE_BOTTOM].x * self.frame_w, landmarks.landmark[LEFT_EYE_BOTTOM].y * self.frame_h])
        eye_left = np.array([landmarks.landmark[LEFT_EYE_LEFT].x * self.frame_w, landmarks.landmark[LEFT_EYE_LEFT].y * self.frame_h])
        eye_right = np.array([landmarks.landmark[LEFT_EYE_RIGHT].x * self.frame_w, landmarks.landmark[LEFT_EYE_RIGHT].y * self.frame_h])
        
        vertical_dist = np.linalg.norm(eye_top - eye_bottom)
        horizontal_dist = np.linalg.norm(eye_left - eye_right)
        
        if horizontal_dist > 0:
            ear = vertical_dist / horizontal_dist
            self._blink_history.append(ear)
            
            if len(self._blink_history) > 0:
                avg_ear = np.mean(self._blink_history)
                self._blinking = avg_ear < 0.2
        else:
            self._blinking = False

    def pupil_left_coords(self) -> Optional[Tuple[int, int]]:
        """Returns left pupil coordinates."""
        if self.pupils_located and self.left_iris:
            return self.left_iris
        return None

    def pupil_right_coords(self) -> Optional[Tuple[int, int]]:
        """Returns right pupil coordinates."""
        if self.pupils_located and self.right_iris:
            return self.right_iris
        return None

    def horizontal_ratio(self) -> Optional[float]:
        if self._blinking:
            self._h_hist.clear()
            return None

        if (
            (not self.pupils_located)
            or self.left_iris is None
            or self.right_iris is None
            or self._left_corner_l is None
            or self._left_corner_r is None
            or self._right_corner_l is None
            or self._right_corner_r is None
        ):
            return None

        l_iris = np.array(self.left_iris, dtype=np.float32)
        r_iris = np.array(self.right_iris, dtype=np.float32)

        l = self._ratio_1d(l_iris, self._left_corner_l, self._left_corner_r)
        r = self._ratio_1d(r_iris, self._right_corner_l, self._right_corner_r)
        if l is None or r is None:
            return None

        raw = self._clamp01((l + r) / 2.0)
        self._h_hist.append(raw)
        return float(np.median(np.asarray(self._h_hist, dtype=float)))

    def vertical_ratio(self) -> Optional[float]:
        if self._blinking:
            self._v_hist.clear()
            return None

        if (
            (not self.pupils_located)
            or self.left_iris is None
            or self.right_iris is None
            or self._left_top is None
            or self._left_bottom is None
            or self._right_top is None
            or self._right_bottom is None
        ):
            return None

        l_iris = np.array(self.left_iris, dtype=np.float32)
        r_iris = np.array(self.right_iris, dtype=np.float32)

        l = self._ratio_1d(l_iris, self._left_top, self._left_bottom)
        r = self._ratio_1d(r_iris, self._right_top, self._right_bottom)
        if l is None or r is None:
            return None

        raw = self._clamp01((l + r) / 2.0)
        self._v_hist.append(raw)
        return float(np.median(np.asarray(self._v_hist, dtype=float)))

    def is_blinking(self) -> bool:
        """Returns True if eyes are closed."""
        return self._blinking

    def is_right(self) -> bool:
        """Returns True if looking right."""
        ratio = self.horizontal_ratio()
        return ratio is not None and ratio <= 0.35

    def is_left(self) -> bool:
        """Returns True if looking left."""
        ratio = self.horizontal_ratio()
        return ratio is not None and ratio >= 0.65

    def is_center(self) -> bool:
        """Returns True if looking center."""
        if self.pupils_located:
            return not self.is_right() and not self.is_left()
        return False

    def annotated_frame(self) -> np.ndarray:
        """Returns frame with pupils and head pose visualized."""
        frame = self.frame.copy()
        
        if self.pupils_located and self.left_iris and self.right_iris:
            color = (0, 255, 0)
            x_left, y_left = int(self.left_iris[0]), int(self.left_iris[1])
            x_right, y_right = int(self.right_iris[0]), int(self.right_iris[1])
            
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color, 2)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color, 2)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color, 2)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color, 2)
        
        if self.head_pose:
            nose_pt = self.head_pose.get('nose_point')
            axis_pts = self.head_pose.get('axis_points')
            if nose_pt and axis_pts:
                p0 = nose_pt
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                for pt, color in zip(axis_pts, colors):
                    cv2.line(frame, p0, pt, color, 2)
            
            angles = self.head_pose.get('angles', {})
            cv2.putText(frame, f"Yaw:{angles.get('yaw',0):.1f}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Pitch:{angles.get('pitch',0):.1f}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Roll:{angles.get('roll',0):.1f}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        return frame