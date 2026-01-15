from __future__ import division
import cv2
from .pupil import Pupil


class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []
        # If blinking_ratio (width/height) exceeds this, treat as blink/closed eye
        self.blinking_ratio_threshold = 3.8
        self.default_threshold = 50

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """Returns the threshold value for the given eye."""
        if side == 0:
            if not self.thresholds_left:
                return self.default_threshold
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            if not self.thresholds_right:
                return self.default_threshold
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """Returns the percentage of space that the iris takes up on the eye surface."""
        if frame is None:
            return 0.0
        frame = frame[5:-5, 5:-5]
        if frame.size == 0:
            return 0.0
        height, width = frame.shape[:2]
        nb_pixels = height * width
        if nb_pixels <= 0:
            return 0.0
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """Calculates the optimal threshold to binarize the frame for the given eye."""
        if eye_frame is None or eye_frame.size == 0:
            return 50
        h, w = eye_frame.shape[:2]
        if h < 10 or w < 10:
            return 50

        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, _iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side, blinking_ratio=None):
        """
        Improves calibration by taking into consideration the given image.
        Skips updates during blinks / invalid eye crops.
        """
        if eye_frame is None or eye_frame.size == 0:
            return
        if blinking_ratio is not None and blinking_ratio > self.blinking_ratio_threshold:
            return

        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)
