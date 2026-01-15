"""Expose MediaPipe-based tracker as the default interface.

Import patterns after this change:
- from gaze_tracking import GazeTracking            # MediaPipe implementation
- from gaze_tracking import GazeTrackingMediaPipe   # Explicit MediaPipe class
"""

from .gaze_tracking_mediapipe import GazeTrackingMediaPipe as GazeTracking
from .gaze_tracking_mediapipe import GazeTrackingMediaPipe
