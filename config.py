"""
Configuration file for the Compensatory Movement Detection System
"""
import os

# Project Structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_VIDEO_DIR = os.path.join(DATA_DIR, 'raw_videos')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_VIDEO_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Movement Classification Labels
MOVEMENT_CLASSES = {
    0: 'correct_form',
    1: 'scapular_winging',
    2: 'hip_sagging',
    3: 'hip_piking',
    4: 'elbow_flaring',
    5: 'trunk_instability'
}

NUM_CLASSES = len(MOVEMENT_CLASSES)

# MediaPipe Configuration
MEDIAPIPE_MODEL_COMPLEXITY = 2  # 0, 1, or 2 (higher = more accurate but slower)
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.7

# MediaPipe Pose Landmarks (33 landmarks)
# Key landmarks for pushup analysis
KEY_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}

# Feature Extraction Configuration
SEQUENCE_LENGTH = 30  # Number of frames to use for temporal analysis
FEATURE_DIM = 78  # Total number of biomechanical features extracted
MIN_FEATURE_DIM = 30  # Minimum acceptable number of features (allows partial extraction)
FEATURE_PADDING_VALUE = 0.0  # Value to use when padding incomplete features
ALLOW_VARIABLE_FEATURES = True  # Allow feature vectors with less than FEATURE_DIM features

# Model Configuration
BILSTM_CONFIG = {
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 7
}

# Data Collection Configuration
VIDEO_PROCESSING = {
    'target_fps': 30,  # Resample videos to this FPS
    'skip_frames': 1,  # Process every Nth frame (1 = process all frames)
    'min_rep_duration': 15,  # Minimum frames for a valid rep
    'max_rep_duration': 90,  # Maximum frames for a valid rep
}

# Rep Counting Configuration
REP_COUNTER = {
    'elbow_angle_threshold_down': 100,  # Degrees - bottom position
    'elbow_angle_threshold_up': 160,    # Degrees - top position
    'hip_height_threshold': 0.15,       # Normalized units for hip movement
    'min_frames_between_reps': 10       # Prevent false counting
}

# Real-time Inference Configuration
INFERENCE_CONFIG = {
    'confidence_threshold': 0.7,  # Minimum confidence to show prediction
    'smoothing_window': 5,        # Frames to smooth predictions
    'display_fps': True,
    'display_rep_count': True,
    'display_feedback': True
}

# Visual Feedback Colors (BGR format for OpenCV)
FEEDBACK_COLORS = {
    'correct_form': (0, 255, 0),        # Green
    'scapular_winging': (0, 165, 255),  # Orange
    'hip_sagging': (0, 0, 255),         # Red
    'hip_piking': (255, 0, 255),        # Magenta
    'elbow_flaring': (0, 255, 255),     # Yellow
    'trunk_instability': (255, 0, 0)    # Blue
}

# Rule-Based Detection Thresholds (for comparison)
RULE_BASED_THRESHOLDS = {
    'scapular_winging': {
        'shoulder_protraction_angle': 30,  # degrees
        'scapular_elevation_diff': 15      # degrees
    },
    'hip_sagging': {
        'hip_angle': 155,  # degrees (should be ~180 for straight body)
        'torso_ankle_angle': 10  # degrees from horizontal
    },
    'hip_piking': {
        'hip_angle': 135,  # degrees (too acute)
        'torso_ankle_angle': -10  # degrees from horizontal
    },
    'elbow_flaring': {
        'elbow_torso_angle': 70,  # degrees (should be <45)
    },
    'trunk_instability': {
        'shoulder_hip_knee_angle_variance': 15,  # degrees
        'lateral_trunk_deviation': 0.05  # normalized units
    }
}

# Data Augmentation (for training)
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'noise_factor': 0.01,
    'rotation_range': 5,  # degrees
    'scale_range': 0.05
}

# Evaluation Metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1-score', 'confusion_matrix']

# Random Seed for Reproducibility
RANDOM_SEED = 42