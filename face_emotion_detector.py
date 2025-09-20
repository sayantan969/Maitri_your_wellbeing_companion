# 1_face_emotion_detector.py
import cv2
from typing import Tuple
from deepface import DeepFace
import numpy as np

class FaceEmotionDetector:
    def __init__(self):
        # The model is loaded automatically by DeepFace on the first run
        print("Face Emotion Detector initialized.")

    def analyze_frame(self, frame: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Analyzes a single video frame for emotion.
        This version is based on the original stable code, with two minimal
        enhancements for performance and accuracy.
        """
        output_frame = frame.copy()
        if frame is None:
            return "N/A", np.zeros((480, 640, 3), dtype=np.uint8)

        dominant_emotion = "N/A"
        label = "No face detected"
        color = (0, 0, 255) # Red for failure

        try:
            # --- ENHANCEMENT 1: Analyze a smaller image for a major speed boost ---
            # This is the most effective way to reduce lag.
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            analyses = DeepFace.analyze(
                img_path=small_frame, # We analyze the small frame
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='ssd' # Using a faster backend also helps
            )
            
            analysis = analyses[0] if isinstance(analyses, list) else analyses
            
            dominant_emotion = analysis['dominant_emotion']
            
            # --- ENHANCEMENT 2: Make the detector more sensitive ---
            # If the model says "neutral", check if another emotion has significant confidence.
            if dominant_emotion == 'neutral':
                for emotion, score in analysis['emotion'].items():
                    if emotion != 'neutral' and score > 25.0: # 25% confidence threshold
                        dominant_emotion = emotion
                        break # Use the first significant emotion found
            
            # Use the final determined emotion for the label
            confidence = analysis['emotion'][dominant_emotion]
            label = f"{dominant_emotion.capitalize()} ({confidence:.1f}%)"
            color = (0, 255, 0) # Green for success

        except Exception:
            # This block remains the same. If anything fails, the defaults are used.
            pass

        # Annotate the full-resolution output frame
        cv2.putText(output_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return dominant_emotion, output_frame

if __name__ == '__main__':
    # This block allows you to test this file independently
    print("Testing FINAL, stable Face Emotion Detector...")
    detector = FaceEmotionDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emotion, annotated_frame = detector.analyze_frame(frame)
        cv2.imshow("Face Emotion Test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()