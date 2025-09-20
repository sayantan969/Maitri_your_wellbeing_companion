import cv2
import numpy as np
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, Label, Frame, scrolledtext, messagebox
from PIL import Image, ImageTk
import queue
import sys
import os
import json

# --- Import Your Custom AI Modules ---
try:
    from face_emotion_detector import FaceEmotionDetector
    from speech_emotion_detector import SpeechEmotionDetector
    from text_sentiment_analyzer import TextSentimentAnalyzer
    from fusion_engine import FusionEngine
    from chat_module import ChatModule, SAMPLE_RATE
    from fatigue_detector import FatigueDetector
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}. Please ensure all module files are in the same directory.")
    sys.exit(1)
except FileNotFoundError as e:
    # This specifically catches the missing dlib model file
    print(f"❌ File not found error: {e}")
    print("Please ensure you have downloaded 'shape_predictor_68_face_landmarks.dat' and placed it in the project folder.")
    sys.exit(1)

# --- Main Application Class ---
class RealTimeEmotionDemo:
    def __init__(self):
        # --- Initialize AI Modules ---
        print("🚀 Initializing MAITRI System...")
        try:
            self.face_detector = FaceEmotionDetector()
            self.speech_detector = SpeechEmotionDetector()
            self.text_analyzer = TextSentimentAnalyzer()
            self.fusion_engine = FusionEngine()
            self.chat_module = ChatModule()
            self.fatigue_detector = FatigueDetector()
            print("✅ All AI modules loaded!")
        except Exception as e:
            print(f"❌ Error loading modules: {e}")
            sys.exit(1)
        
        # --- Initialize GUI ---
        self.root = tk.Tk()
        self.root.title("MAITRI - Advanced Emotion Analysis & Counseling System")
        self.root.geometry("1400x900") # Adjusted height
        self.root.configure(bg='#2C3E50')
        
        # --- Video Capture ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # --- State Variables ---
        self.current_emotions = {'face': 'N/A', 'speech': 'N/A', 'text': 'N/A', 'fused': 'N/A'}
        self.fatigue_alert_threshold = 5  # Alert if total events > 5
        self.fatigue_alert_shown_this_session = False
        self.is_running = True
        self.analysis_active = False # Controls the conversation loop
        
        self.setup_gui()
        self.start_video_thread()
        
    def setup_gui(self):
        """Create the entire GUI layout."""
        # --- Main Structure ---
        title_frame = Frame(self.root, bg='#34495E', height=70)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        Label(title_frame, text="🚀 MAITRI - Advanced Emotion Analysis & Counseling System",
              font=('Arial', 22, 'bold'), fg='white', bg='#34495E').pack(expand=True)
        
        main_frame = Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # --- Left Side (Video, Analysis, Controls) ---
        left_frame = Frame(main_frame, bg='#34495E', width=680)
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        Label(left_frame, text="📹 Live Video Feed", font=('Arial', 16, 'bold'), fg='white', bg='#34495E').pack(pady=5)
        self.video_label = Label(left_frame, bg='black')
        self.video_label.pack(pady=5)

        panels_frame = Frame(left_frame, bg='#34495E')
        panels_frame.pack(fill='x', pady=10, padx=10)

        # Emotion Analysis Panel
        emotion_frame = Frame(panels_frame, bg='#2C3E50', bd=2, relief='ridge')
        emotion_frame.pack(side='left', fill='x', expand=True, padx=5)
        Label(emotion_frame, text="🧠 Live Emotion Analysis", font=('Arial', 14, 'bold'), fg='white', bg='#2C3E50').grid(row=0, column=0, columnspan=2, pady=5)
        self.emotion_labels = {}
        for i, (text, key) in enumerate([('Face', 'face'), ('Speech', 'speech'), ('Text', 'text'), ('Fused', 'fused')]):
            Label(emotion_frame, text=f"{text}:", font=('Arial', 12), fg='white', bg='#2C3E50').grid(row=i+1, column=0, sticky='w', padx=5, pady=2)
            self.emotion_labels[key] = Label(emotion_frame, text="N/A", font=('Arial', 12, 'bold'), fg='#3498DB', bg='#2C3E50')
            self.emotion_labels[key].grid(row=i+1, column=1, sticky='w', padx=5, pady=2)

        # Fatigue Analysis Panel
        fatigue_frame = Frame(panels_frame, bg='#2C3E50', bd=2, relief='ridge')
        fatigue_frame.pack(side='right', fill='x', expand=True, padx=5)
        Label(fatigue_frame, text="😴 Live Fatigue Analysis", font=('Arial', 14, 'bold'), fg='white', bg='#2C3E50').grid(row=0, column=0, columnspan=2, pady=5)
        self.fatigue_labels = {}
        for i, (text, key) in enumerate([('Yawns', 'yawns'), ('Drowsy Events', 'drowsy')]):
            Label(fatigue_frame, text=f"{text}:", font=('Arial', 12), fg='white', bg='#2C3E50').grid(row=i+1, column=0, sticky='w', padx=5, pady=2)
            self.fatigue_labels[key] = Label(fatigue_frame, text="0", font=('Arial', 12, 'bold'), fg='#E67E22', bg='#2C3E50')
            self.fatigue_labels[key].grid(row=i+1, column=1, sticky='w', padx=5, pady=2)
        
        # Control Buttons
        button_frame = Frame(left_frame, bg='#34495E')
        button_frame.pack(fill='x', pady=10, side='bottom')
        self.start_btn = tk.Button(button_frame, text="▶️ Start Analysis", command=self.toggle_analysis, bg='#27AE60', fg='white', font=('Arial', 12, 'bold'), relief='raised')
        self.start_btn.pack(fill='x', pady=3, padx=20)
        self.report_btn = tk.Button(button_frame, text="📄 Generate Fatigue Report", command=self.generate_fatigue_report, bg='#9B59B6', fg='white', font=('Arial', 12, 'bold'), relief='raised')
        self.report_btn.pack(fill='x', pady=3, padx=20)
        quit_btn = tk.Button(button_frame, text="❌ Quit", command=self.quit_app, bg='#E74C3C', fg='white', font=('Arial', 12, 'bold'), relief='raised')
        quit_btn.pack(fill='x', pady=3, padx=20)
        
        # --- Right Side (Chat Interface) ---
        right_frame = Frame(main_frame, bg='#34495E', width=700, height=750)
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        right_frame.pack_propagate(False)
        self.chat_history = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, state='disabled', font=('Arial', 12), bg='#ECF0F1', fg='#2C3E50', padx=10, pady=10)
        self.chat_history.pack(pady=10, padx=10, expand=True, fill='both')
        # Chat bubble styling
        self.chat_history.tag_config('user', justify='right', rmargin=10, background='#DCF8C6', wrap='word', spacing3=10, relief='raised', borderwidth=1)
        self.chat_history.tag_config('assistant', justify='left', lmargin1=10, lmargin2=10, background='#FFFFFF', wrap='word', spacing3=10, relief='raised', borderwidth=1)
        self.chat_history.tag_config('system', justify='center', foreground='gray', font=('Arial', 10, 'italic'), spacing3=10)

    def start_video_thread(self):
        """Starts the thread that handles video capture and live analysis."""
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
    def video_loop(self):
        """Continuously captures video frames and runs live detectors."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # --- Perform live analysis on frame copies ---
            emotion_frame, fatigue_frame = frame.copy(), frame.copy()
            
            try:
                emotion, emotion_frame = self.face_detector.analyze_frame(emotion_frame)
                self.current_emotions['face'] = emotion.lower()
            except Exception:
                self.current_emotions['face'] = "error"
            
            try:
                fatigue_frame, alert = self.fatigue_detector.process_frame(fatigue_frame)
                if alert and self.analysis_active:
                    total_events = self.fatigue_detector.total_yawns + self.fatigue_detector.total_drowsy_events
                    if total_events > self.fatigue_alert_threshold and not self.fatigue_alert_shown_this_session:
                        self.root.after(0, self.show_fatigue_warning)
                        self.fatigue_alert_shown_this_session = True
            except Exception as e:
                print(f"Fatigue analysis error: {e}")

            # --- Display Video Feed ---
            final_frame = cv2.addWeighted(emotion_frame, 0.5, fatigue_frame, 0.5, 0)
            self.root.after(0, self.update_live_displays)
            
            frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(frame_rgb, (640, 480))))
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            
            time.sleep(0.03) # Limit to ~30 FPS

    def update_live_displays(self):
        """Update GUI labels for live analysis."""
        self.emotion_labels['face'].config(text=self.current_emotions.get('face', "N/A").upper())
        self.fatigue_labels['yawns'].config(text=str(self.fatigue_detector.total_yawns))
        self.fatigue_labels['drowsy'].config(text=str(self.fatigue_detector.total_drowsy_events))

    def update_turn_displays(self):
        """Update GUI labels after a full conversation turn."""
        self.emotion_labels['speech'].config(text=self.current_emotions.get('speech', "N/A").upper())
        self.emotion_labels['text'].config(text=self.current_emotions.get('text', "N/A").upper())
        self.emotion_labels['fused'].config(text=self.current_emotions.get('fused', "N/A").upper())

    def add_message_to_chat(self, sender, message):
        """Adds a formatted message to the chat window on the main thread."""
        def task():
            self.chat_history.config(state='normal')
            self.chat_history.insert(tk.END, f"{message}\n\n", sender)
            self.chat_history.config(state='disabled')
            self.chat_history.yview(tk.END)
        self.root.after(0, task)

    def toggle_analysis(self):
        """Starts or stops the conversation loop."""
        self.analysis_active = not self.analysis_active
        if self.analysis_active:
            self.start_btn.config(text="⏸️ Pause Analysis", bg='#E67E22')
            self.add_message_to_chat('system', 'Session started. Please speak now...')
            self.fatigue_detector.total_yawns = 0
            self.fatigue_detector.total_drowsy_events = 0
            self.fatigue_alert_shown_this_session = False
            threading.Thread(target=self.conversation_manager_loop, daemon=True).start()
        else:
            self.start_btn.config(text="▶️ Start Analysis", bg='#27AE60')
            self.add_message_to_chat('system', 'Session paused.')

    def conversation_manager_loop(self):
        """Manages the sequence of a conversation while analysis is active."""
        while self.analysis_active:
            self.run_conversation_step()
            time.sleep(1) # Brief pause between turns

    def run_conversation_step(self):
        """Executes one full turn: record -> analyze -> respond."""
        # --- 1. Audio Recording & Transcription ---
        self.add_message_to_chat('system', "Listening...")
        audio_data = self.chat_module._record_audio()
        if not self.analysis_active: return
        self.add_message_to_chat('system', "Transcribing audio...")
        user_input = self.chat_module.speech_to_text_from_data(audio_data)
        
        if not user_input.strip():
            self.add_message_to_chat('system', "No speech detected. Listening again...")
            return
        self.add_message_to_chat('user', user_input)
        
        # --- 2. Multi-Modal Emotion Analysis ---
        self.add_message_to_chat('system', "Fusing emotions...")
        try:
            speech_emo = self.speech_detector.analyze_audio((SAMPLE_RATE, audio_data)).lower()
            text_emo = self.text_analyzer.analyze_text(user_input).lower()
            fused_emo = self.fusion_engine.fuse_emotions(self.current_emotions['face'], speech_emo, text_emo).lower()
            self.current_emotions.update({'speech': speech_emo, 'text': text_emo, 'fused': fused_emo})
        except Exception as e:
            print(f"Emotion analysis pipeline error: {e}")
            self.current_emotions['fused'] = self.current_emotions['face'] # Fallback
        
        self.root.after(0, self.update_turn_displays)
        
        # --- 3. Generate AI Response ---
        self.add_message_to_chat('system', f"Felt Emotion: {self.current_emotions['fused'].upper()}. Generating response...")
        try:
            response = self.chat_module.get_response(self.current_emotions['fused'], user_input)
            self.add_message_to_chat('assistant', response)
        except Exception as e:
            error_msg = f"Error generating AI response: {e}"
            self.add_message_to_chat('system', error_msg)
            print(error_msg)

    def generate_fatigue_report(self):
        """Creates a pop-up window with fatigue statistics for the current session."""
        report = (
            f"Fatigue Analysis Report (Current Session)\n\n"
            f"Total Yawns Detected: {self.fatigue_detector.total_yawns}\n"
            f"Total Drowsiness Events: {self.fatigue_detector.total_drowsy_events}\n"
        )
        messagebox.showinfo("Fatigue Report", report)

    def show_fatigue_warning(self):
        """Creates a pop-up warning if fatigue threshold is crossed."""
        messagebox.showwarning("High Fatigue Alert", "A high level of fatigue has been detected! Please consider taking a break.")

    def quit_app(self):
        """Handles a clean shutdown of the application."""
        print("🛑 Shutting down MAITRI...")
        self.is_running = False
        self.analysis_active = False
        time.sleep(0.5)
        if self.cap: self.cap.release()
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Starts the Tkinter application."""
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.mainloop()

if __name__ == "__main__":
    app = RealTimeEmotionDemo()
    app.run()
