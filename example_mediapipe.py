"""
Demo of MediaPipe-based gaze tracking.
Run with: python example_mediapipe.py
Provides same interface as example.py but uses MediaPipe instead of dlib.
"""

import cv2
from gaze_tracking.gaze_tracking_mediapipe import GazeTrackingMediaPipe

gaze = GazeTrackingMediaPipe()
webcam = cv2.VideoCapture(0)

print("MediaPipe Gaze Tracking Demo")
print("Press 'q' to quit")

frame_count = 0
while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    # Process frame
    gaze.refresh(frame)
    frame_annotated = gaze.annotated_frame()
    
    # Gaze direction text
    text = ""
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"
    
    cv2.putText(frame_annotated, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    
    # Pupil coordinates
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    
    cv2.putText(frame_annotated, "Left pupil:  " + str(left_pupil), (90, 130), 
               cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame_annotated, "Right pupil: " + str(right_pupil), (90, 165), 
               cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    
    # Gaze ratios
    h_ratio = gaze.horizontal_ratio()
    v_ratio = gaze.vertical_ratio()
    h_text = f"H ratio: {h_ratio:.2f}" if h_ratio else "H ratio: N/A"
    v_text = f"V ratio: {v_ratio:.2f}" if v_ratio else "V ratio: N/A"
    
    cv2.putText(frame_annotated, h_text, (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame_annotated, v_text, (90, 235), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    
    # Frame counter
    frame_count += 1
    cv2.putText(frame_annotated, f"Frame: {frame_count}", (90, 270), 
               cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    
    cv2.imshow("MediaPipe Gaze Tracking Demo", frame_annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
print(f"Demo ended. Processed {frame_count} frames.")