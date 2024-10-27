import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import os
from pathlib import Path
import sys

from emotion_detector import EmotionDetector
from music_player import MusicPlayer
from config import *

class EmotionMusicPlayerApp:
    def __init__(self):
        try:
            # Initialize main window first
            self.root = tk.Tk()
            self.root.title("Emotion-Based Music Player")
            self.root.geometry("800x600")
            
            # Initialize components
            self.setup_components()
            self.setup_gui()
            
            # Start video capture
            self.is_running = True
            self.start_video_capture()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            if hasattr(self, 'root'):
                self.root.destroy()
            sys.exit(1)
        
    def setup_components(self):
        """Initialize emotion detector and music player"""
        try:
            print("Initializing components...")
            # Initialize emotion detector
            self.emotion_detector = EmotionDetector()
            
            # Initialize music player
            self.music_player = MusicPlayer()
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open video capture")
            
            print("Components initialized successfully")
            
        except Exception as e:
            raise Exception(f"Error setting up components: {str(e)}")
            
    def setup_gui(self):
        """Set up the GUI elements"""
        try:
            # Create main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(expand=True, fill='both')
            
            # Video frame
            self.video_label = ttk.Label(main_frame)
            self.video_label.grid(row=0, column=0, padx=5, pady=5)
            
            # Information frame
            info_frame = ttk.Frame(main_frame)
            info_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
            
            # Status label
            self.status_label = ttk.Label(info_frame, text="Status: Initializing...")
            self.status_label.pack(pady=5)
            
            # Emotion label
            self.emotion_label = ttk.Label(info_frame, text="Detected Emotion: ")
            self.emotion_label.pack(pady=5)
            
            # Valence/Arousal label
            self.va_label = ttk.Label(info_frame, text="Valence/Arousal: ")
            self.va_label.pack(pady=5)
            
            # Song label
            self.song_label = ttk.Label(info_frame, text="Currently Playing: None")
            self.song_label.pack(pady=5)
            
            # Control buttons
            ttk.Button(info_frame, text="Stop Music", 
                      command=self.stop_music).pack(pady=5)
            ttk.Button(info_frame, text="Exit", 
                      command=self.cleanup).pack(pady=5)
            
        except Exception as e:
            raise Exception(f"Error setting up GUI: {str(e)}")
        
    def start_video_capture(self):
        """Start the video capture thread"""
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()
        
    def update_frame(self):
        """Update the video frame and process emotions"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Detect emotion
                emotion, valence, arousal = self.emotion_detector.detect_emotion(frame)
                
                if emotion:
                    # Update status
                    self.status_label.config(text="Status: Running")
                    
                    # Update labels
                    self.emotion_label.config(text=f"Detected Emotion: {emotion}")
                    self.va_label.config(
                        text=f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")
                    
                    # Find and play matching music
                    matching_song = self.music_player.find_matching_song(
                        valence, arousal)
                    
                    if matching_song is not None:
                        if self.music_player.play_song(matching_song['audio_path']):
                            self.song_label.config(
                                text=f"Playing: Song {matching_song['song_id']}")
                
                # Update video display
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                img = img.resize((400, 300))  # Resize for display
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
            self.root.update()
            
    def stop_music(self):
        """Stop the currently playing music"""
        self.music_player.stop_song()
        self.song_label.config(text="Currently Playing: None")
        
    def cleanup(self):
        """Clean up resources before closing"""
        self.is_running = False
        self.music_player.stop_song()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = EmotionMusicPlayerApp()
        app.run()
    except Exception as e:
        print(f"Application failed to start: {str(e)}")
        sys.exit(1)