import numpy as np
import cv2
import math


class HeadPose(object):
    """
    Estimate head pose (yaw, pitch, roll) from dlib 68-point landmarks
    using a 3D model and cv2.solvePnP.
    """

    def __init__(self):
        # 3D model points in mm (generic face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=float)

    @staticmethod
    def _landmark_to_np(landmarks, indices):
        pts = []
        for i in indices:
            p = landmarks.part(i)
            pts.append((p.x, p.y))
        return np.array(pts, dtype=float)

    def estimate(self, landmarks, frame):
        """
        landmarks: dlib.full_object_detection
        frame: numpy.ndarray (color or gray) used only for size
        Returns:
          dict {
            'rvec': rvec,
            'tvec': tvec,
            'angles': {'yaw':..., 'pitch':..., 'roll':...},
            'nose_point': (x,y),
            'axis_points': [(x1,y1),(x2,y2),(x3,y3)]
          }
        and draws axis points projection based on the nose.
        """
        # dlib indices for the chosen image points
        IMAGE_INDICES = [30, 8, 36, 45, 48, 54]  # nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
        image_points = self._landmark_to_np(landmarks, IMAGE_INDICES)

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=float)
        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        success, rvec, tvec = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None

        # Project 3 axis points (for visualization)
        axis_3d = np.float32([[100.0, 0.0, 0.0],
                              [0.0, 100.0, 0.0],
                              [0.0, 0.0, 100.0]])
        nose_3d = np.float32([[0.0, 0.0, 0.0]])
        nose_point_2d, _ = cv2.projectPoints(nose_3d, rvec, tvec, camera_matrix, dist_coeffs)
        axis_points_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

        nose_point = (int(nose_point_2d[0][0][0]), int(nose_point_2d[0][0][1]))
        axis_points = [(int(p[0][0]), int(p[0][1])) for p in axis_points_2d]

        # Rotation vector -> rotation matrix -> Euler angles
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

        # Convert to degrees: pitch (x), yaw (y), roll (z)
        angles = {
            'pitch': math.degrees(x),
            'yaw': math.degrees(y),
            'roll': math.degrees(z)
        }

        return {
            'rvec': rvec,
            'tvec': tvec,
            'angles': angles,
            'nose_point': nose_point,
            'axis_points': axis_points,
            'image_points': image_points
        }

    @staticmethod
    def draw_axes(frame, nose_point, axis_points):
        """
        Draw simple RGB axes on the frame:
          X -> red, Y -> green, Z -> blue
        """
        if nose_point is None:
            return frame
        p0 = nose_point
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for pt, color in zip(axis_points, colors):
            cv2.line(frame, p0, pt, color, 2)
        return frame