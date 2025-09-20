import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import os

class FatigueDetector:
    def __init__(self, ear_thresh=0.25, ear_consec_frames=20, mar_thresh=0.5, mar_consec_frames=15):
        # --- Constants and Thresholds ---
        self.EAR_THRESHOLD = ear_thresh
        self.EAR_CONSEC_FRAMES = ear_consec_frames
        self.MAR_THRESHOLD = mar_thresh
        self.MAR_CONSEC_FRAMES = mar_consec_frames

        # --- Frame Counters for current event ---
        self.eye_frame_counter = 0
        self.yawn_frame_counter = 0

        # --- Total Session Counters ---
        self.total_yawns = 0
        self.total_drowsy_events = 0
        
        # --- Model Initialization ---
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Error: The dlib model file '{model_path}' was not found. Please download it and place it in the same directory as the script.")
            
        print("⏳ [INFO] Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

        # --- Landmark indices ---
        self.lStart, self.lEnd = (42, 48)
        self.rStart, self.rEnd = (36, 42)
        self.mStart, self.mEnd = (60, 68)
        print("✅ [INFO] Fatigue Detector Initialized.")

    def _eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[6])
        B = dist.euclidean(mouth[0], mouth[4])
        return A / B

    def process_frame(self, frame):
        alert = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if rects:
            rect = rects[0]
            shape = self.predictor(gray, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[self.mStart:self.mEnd]
            
            ear = (self._eye_aspect_ratio(leftEye) + self._eye_aspect_ratio(rightEye)) / 2.0
            mar = self._mouth_aspect_ratio(mouth)

            # --- Draw contours ---
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            # --- Check for eye closure (drowsiness) ---
            if ear < self.EAR_THRESHOLD:
                self.eye_frame_counter += 1
                if self.eye_frame_counter >= self.EAR_CONSEC_FRAMES:
                    self.total_drowsy_events += 1
                    self.eye_frame_counter = 0 # Reset after detecting
                    alert = True
            else:
                self.eye_frame_counter = 0

            # --- Check for yawns ---
            if mar > self.MAR_THRESHOLD:
                self.yawn_frame_counter += 1
            else:
                if self.yawn_frame_counter >= self.MAR_CONSEC_FRAMES:
                    self.total_yawns += 1
                    self.yawn_frame_counter = 0 # Reset after detecting
                    alert = True
                self.yawn_frame_counter = 0 # Reset if mouth closes
            
            if alert:
                 cv2.putText(frame, "FATIGUE ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, alert
