from fer import FER
import cv2
import numpy as np

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detector"""
        self.detector = FER(mtcnn=True)
        
    def detect_emotion(self, frame):
        """
        Detect emotions in a frame
        
        Args:
            frame: CV2 image frame
            
        Returns:
            tuple: (dominant_emotion, valence, arousal)
        """
        try:
            # Detect emotions
            emotions = self.detector.detect_emotions(frame)
            
            if emotions:
                # Get the dominant emotion
                emotions_dict = emotions[0]['emotions']
                dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])[0]
                
                # Convert emotion to valence/arousal
                valence, arousal = self.emotion_to_valence_arousal(dominant_emotion)
                
                return dominant_emotion, valence, arousal
            
            return None, 0, 0
            
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return None, 0, 0
    
    def emotion_to_valence_arousal(self, emotion):
        """Convert detected emotion to valence and arousal values"""
        emotion_map = {
            'happy': (0.8, 0.8),     # High valence, high arousal
            'sad': (-0.8, -0.4),     # Low valence, low arousal
            'angry': (-0.7, 0.8),    # Low valence, high arousal
            'neutral': (0.0, 0.0),   # Neutral valence, neutral arousal
            'fear': (-0.8, 0.5),     # Low valence, high arousal
            'surprise': (0.4, 0.8),  # Medium-high valence, high arousal
            'disgust': (-0.8, 0.1)   # Low valence, medium arousal
        }
        return emotion_map.get(emotion, (0.0, 0.0))