# 4_fusion_engine.py
from collections import Counter

class FusionEngine:
    def __init__(self):
        # Mapping from model outputs to a standardized set of emotions
        self.emotion_map = {
            'happy': 'Happy',
            'sad': 'Sad',
            'angry': 'Angry',
            'fear': 'Fear',
            'surprise': 'Surprise',
            'disgust': 'Disgust',
            'neutral': 'Neutral',
            'Positive': 'Happy',
            'Negative': 'Sad',
            'Neutral': 'Neutral'
        }
        print("Fusion Engine initialized.")

    def fuse_emotions(self, face_emo: str, speech_emo: str, text_emo: str) -> str:
        """
        Combines multimodal inputs using majority voting.

        Args:
            face_emo: Emotion detected from face.
            speech_emo: Emotion detected from speech.
            text_emo: Sentiment detected from text.

        Returns:
            The final fused emotion string.
        """
        # Standardize the emotion labels
        face_emo_std = self.emotion_map.get(face_emo.lower(), "N/A")
        speech_emo_std = self.emotion_map.get(speech_emo.lower(), "N/A")
        text_emo_std = self.emotion_map.get(text_emo.lower(), "N/A")
        
        # Collect valid (non-"N/A") detections
        detected_emotions = [e for e in [face_emo_std, speech_emo_std, text_emo_std] if e != "N/A"]

        if not detected_emotions:
            return "N/A"

        # Use majority voting
        most_common = Counter(detected_emotions).most_common(1)[0]
        
        # If there's a tie, most_common returns one of them arbitrarily. This is fine for our case.
        fused_emotion = most_common[0]
        
        return fused_emotion

if __name__ == '__main__':
    # This block allows you to test this file independently
    print("Testing Fusion Engine...")
    engine = FusionEngine()
    
    # Test case 1: Agreement
    result1 = engine.fuse_emotions('happy', 'Happy', 'Positive')
    print(f"Inputs: ('happy', 'Happy', 'Positive') -> Fused: {result1}")
    
    # Test case 2: Disagreement
    result2 = engine.fuse_emotions('angry', 'Sad', 'Negative')
    print(f"Inputs: ('angry', 'Sad', 'Negative') -> Fused: {result2}")
    
    # Test case 3: One modality missing
    result3 = engine.fuse_emotions('neutral', 'N/A', 'Neutral')
    print(f"Inputs: ('neutral', 'N/A', 'Neutral') -> Fused: {result3}")