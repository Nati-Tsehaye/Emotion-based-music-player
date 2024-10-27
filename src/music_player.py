import numpy as np
import pandas as pd
import tensorflow as tf
import os
from pygame import mixer
from pathlib import Path
import pickle
from config import *
import time
class MusicPlayer:
    def __init__(self):
        """Initialize the music player with the configured paths"""
        try:
            # Initialize audio with fallback options
            self._init_audio()
            
            # Set initial volume
            self.volume = 0.5
            mixer.music.set_volume(self.volume)
            
            # Load model and scaler
            self.load_models()
            
            # Load dataset
            self.load_dataset()
            
            self.current_song = None
            self.is_playing = False
            self.last_play_time = 0
            self.min_play_interval = 1.0  # Minimum time (in seconds) between song changes
            
        except Exception as e:
            raise Exception(f"Music Player initialization error: {str(e)}")

    def _init_audio(self):
        """Initialize audio with fallback options"""
        try:
            # First try with default settings
            mixer.init()
        except Exception as e1:
            try:
                # If default fails, try with specific settings
                mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            except Exception as e2:
                try:
                    # Last resort: try with basic settings
                    mixer.quit()  # Ensure mixer is fully stopped
                    mixer.init(frequency=22050, size=-16, channels=1, buffer=2048)
                except Exception as e3:
                    raise Exception(f"Could not initialize audio system: {str(e3)}")
        
        print("Audio system initialized successfully")
        

    def load_models(self):
        """Load the music emotion model and scaler"""
        try:
            print("Loading model from:", MODEL_PATH)
            
            # Load TensorFlow model with custom_objects
            self.model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'loss': tf.keras.losses.MeanSquaredError(),
                    'metric': tf.keras.metrics.MeanAbsoluteError()
                }
            )
            
            print("Loading scaler from:", SCALER_PATH)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("Loading feature columns from:", FEATURE_COLUMNS_PATH)
            self.feature_columns = np.load(FEATURE_COLUMNS_PATH, allow_pickle=True)
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Detailed error loading models: {str(e)}")
            raise Exception(f"Error loading models: {str(e)}")

    def load_dataset(self):
        """Load and prepare the music dataset"""
        try:
            # Load features
            features_list = []
            missing_files = []
            print("Loading features from:", FEATURES_PATH)
            
            for song_id in range(2, 2058):
                feature_file = os.path.join(FEATURES_PATH, f'{song_id}.csv')
                if not os.path.isfile(feature_file):
                    missing_files.append(song_id)
                    continue
                    
                try:
                    features = pd.read_csv(feature_file, sep=';', header=None)
                    features = features.dropna(axis=1, how='all')
                    features = features.apply(pd.to_numeric, errors='coerce')
                    
                    mean_features = features.mean(skipna=True).to_frame().T
                    mean_features['song_id'] = song_id
                    mean_features['audio_path'] = os.path.join(AUDIO_PATH, f'{song_id}.mp3')
                    
                    features_list.append(mean_features)
                    
                except Exception as e:
                    print(f"Error processing song {song_id}: {str(e)}")
                    missing_files.append(song_id)
                    continue
            
            features_df = pd.concat(features_list, ignore_index=True)
            
            # Load annotations
            print("Loading annotations...")
            # Load first set (1-2000)
            labels_1 = pd.read_csv(ANNOTATIONS_1_2000)
            labels_1.columns = [col.strip().lower() for col in labels_1.columns]
            labels_1 = labels_1[['song_id', 'valence_mean', 'arousal_mean']]
            labels_1 = labels_1.rename(columns={
                'valence_mean': 'valence_average',
                'arousal_mean': 'arousal_average'
            })
            
            # Load second set (2000-2058)
            labels_2 = pd.read_csv(ANNOTATIONS_2000_2058)
            labels_2.columns = [col.strip().lower() for col in labels_2.columns]
            labels_2 = labels_2[['song_id', 'valence_mean', 'arousal_mean']]
            labels_2 = labels_2.rename(columns={
                'valence_mean': 'valence_average',
                'arousal_mean': 'arousal_average'
            })
            
            # Combine labels
            labels_df = pd.concat([labels_1, labels_2], ignore_index=True)
            labels_df = labels_df.groupby('song_id').agg({
                'valence_average': 'mean',
                'arousal_average': 'mean'
            }).reset_index()
            
            # Merge features and labels
            self.merged_df = pd.merge(features_df, labels_df, on='song_id')
            
            # Handle missing values
            numeric_columns = self.merged_df.select_dtypes(include=np.number).columns
            self.merged_df[numeric_columns] = self.merged_df[numeric_columns].fillna(
                self.merged_df[numeric_columns].mean())
                
            print(f"Dataset loaded successfully. Total songs: {len(self.merged_df)}")
            
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
            
    def find_matching_song(self, valence, arousal):
        """Find a matching song for given valence and arousal values"""
        try:
            # Calculate Euclidean distance between target and all songs
            self.merged_df['distance'] = np.sqrt(
                (self.merged_df['valence_average'] - valence) ** 2 +
                (self.merged_df['arousal_average'] - arousal) ** 2
            )
            
            # Find the song with minimum distance
            closest_song = self.merged_df.loc[self.merged_df['distance'].idxmin()]
            
            # Check if the audio file exists
            if os.path.exists(closest_song['audio_path']):
                return closest_song
            else:
                print(f"Audio file not found: {closest_song['audio_path']}")
                # Try to find the next closest song
                self.merged_df = self.merged_df[self.merged_df.index != closest_song.name]
                return self.find_matching_song(valence, arousal)
                
        except Exception as e:
            print(f"Error finding matching song: {str(e)}")
            return None
            
    def play_song(self, song_path):
        """Play a song from the given path with enhanced playback control"""
        try:
            # Check if enough time has passed since last playback
            current_time = time.time()
            if current_time - self.last_play_time < self.min_play_interval:
                print("Please wait before playing another song")
                return False

            # Check if it's the same song that's already playing
            if self.current_song == song_path and self.is_playing:
                print("This song is already playing")
                return True

            if not os.path.exists(song_path):
                print(f"Error: Song file not found at {song_path}")
                return False
                
            # Check file size
            if os.path.getsize(song_path) == 0:
                print(f"Error: Song file is empty: {song_path}")
                return False
                
            # Stop current playback if any
            if self.is_playing:
                self.stop_song()
                time.sleep(0.1)  # Small delay to ensure clean transition
                
            try:
                mixer.music.load(song_path)
                mixer.music.set_volume(self.volume)
                mixer.music.play()
                self.current_song = song_path
                self.is_playing = True
                self.last_play_time = current_time
                print(f"Successfully playing: {song_path}")
                print(f"Current volume: {self.volume:.2f}")
                return True
                
            except Exception as e:
                print(f"Error playing file {song_path}: {str(e)}")
                return False
                
        except Exception as e:
            print(f"Error in play_song: {str(e)}")
            return False

    def stop_song(self):
        """Stop the currently playing song"""
        try:
            if self.is_playing:
                mixer.music.stop()
                self.current_song = None
                self.is_playing = False
                print("Playback stopped")
        except Exception as e:
            print(f"Error stopping song: {str(e)}")

    def set_volume(self, volume):
        """Set the volume (0.0 to 1.0)"""
        try:
            self.volume = max(0.0, min(1.0, volume))
            mixer.music.set_volume(self.volume)
            print(f"Volume set to: {self.volume:.2f}")
            return True
        except Exception as e:
            print(f"Error setting volume: {str(e)}")
            return False

    def get_playback_status(self):
        """Get current playback status"""
        return {
            'is_playing': self.is_playing,
            'current_song': self.current_song,
            'volume': self.volume
        }

# Example of how to use the MusicPlayer class
if __name__ == "__main__":
    try:
        player = MusicPlayer()
        print("MusicPlayer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize MusicPlayer: {str(e)}")