# 2_speech_emotion_detector.py
import os
from typing import Tuple
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel
import librosa
import numpy as np

class SpeechEmotionDetector:
    def __init__(self):
        self.MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "local_wav2vec2_speech_model")
        self.SAMPLING_RATE = 16000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Initializing Speech Emotion Detector...")
        self.processor, self.model = self._load_model()
        self.model.eval()
        print("Speech Emotion Detector initialized.")

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        def forward(self, features):
            x = self.dropout(features)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = SpeechEmotionDetector.RegressionHead(config)
            self.init_weights()
        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            hidden_states = torch.mean(hidden_states, dim=1)
            logits = self.classifier(hidden_states)
            return logits

    def _load_model(self):
        if os.path.exists(self.LOCAL_MODEL_PATH):
            print(f"Loading speech model from local path: {self.LOCAL_MODEL_PATH}")
            processor = Wav2Vec2Processor.from_pretrained(self.LOCAL_MODEL_PATH)
            model = self.EmotionModel.from_pretrained(self.LOCAL_MODEL_PATH).to(self.device)
        else:
            print("Downloading speech model (one-time operation)...")
            processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
            model = self.EmotionModel.from_pretrained(self.MODEL_NAME).to(self.device)
            processor.save_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(self.LOCAL_MODEL_PATH)
            print(f"Speech model saved to {self.LOCAL_MODEL_PATH}")
        return processor, model

    def _avd_to_label(self, arousal, dominance, valence, v_th=0.55, a_th=0.55, d_th=0.5):
        if valence >= v_th and arousal >= a_th: return "happy"
        if valence < v_th and arousal < 0.4: return "sad"
        if valence < v_th and arousal >= a_th: return "angry" if dominance >= d_th else "fear"
        if abs(valence - 0.5) < 0.2 and arousal >= a_th: return "surprise"
        if valence < v_th and 0.35 <= arousal < a_th: return "disgust"
        return "neutral"

    def analyze_audio(self, audio_data: Tuple[int, np.ndarray]) -> str:
        """
        Analyzes a raw audio chunk for emotion.

        Args:
            audio_data: A tuple from Gradio's microphone component (sample_rate, numpy_array).

        Returns:
            The detected emotion label (str).
        """
        if audio_data is None:
            return "N/A"
            
        sr, audio = audio_data
        if audio is None or len(audio) == 0:
            return "N/A"
            
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Resample if necessary
        if sr != self.SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLING_RATE)
        
        inputs = self.processor(audio, sampling_rate=self.SAMPLING_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device))
        
        avd = logits.cpu().numpy()[0]
        label = self._avd_to_label(avd[0], avd[1], avd[2])
        return label.capitalize()

if __name__ == '__main__':
    # This block allows you to test this file independently
    print("Testing Speech Emotion Detector...")
    detector = SpeechEmotionDetector()
    # Create a dummy audio signal for testing
    sr = 44100
    duration = 3
    dummy_audio = np.random.randn(sr * duration)
    result = detector.analyze_audio((sr, dummy_audio))
    print(f"Test analysis complete. Detected emotion: {result}")