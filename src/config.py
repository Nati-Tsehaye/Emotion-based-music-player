import os

# Base paths
BASE_DIR = r"C:\Users\Nati\Desktop\emotion_music_player"

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_music_model (1).h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "feature_scaler (1).pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns (1).npy")

# Data paths
DEAM_BASE = os.path.join(BASE_DIR, "data", "DEAM_audio")
AUDIO_PATH = os.path.join(DEAM_BASE, "MEMD_audio")
FEATURES_PATH = os.path.join(DEAM_BASE, "features")
ANNOTATIONS_PATH = os.path.join(DEAM_BASE, "annotations")

# Annotation files
ANNOTATIONS_1_2000 = os.path.join(ANNOTATIONS_PATH, "static_annotations_averaged_songs_1_2000.csv")
ANNOTATIONS_2000_2058 = os.path.join(ANNOTATIONS_PATH, "static_annotations_averaged_songs_2000_2058.csv")