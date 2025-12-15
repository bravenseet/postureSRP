"""
Utility functions for the Compensatory Movement Detection System
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
import config


def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """
    Calculate angle between three points (point2 is the vertex)

    Args:
        point1, point2, point3: 3D coordinates [x, y, z]

    Returns:
        Angle in degrees
    """
    vector1 = point1 - point2
    vector2 = point3 - point2

    # Handle zero vectors
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    vector1_normalized = vector1 / norm1
    vector2_normalized = vector2 / norm2

    dot_product = np.dot(vector1_normalized, vector2_normalized)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points

    Args:
        point1, point2: 3D coordinates [x, y, z]

    Returns:
        Distance
    """
    return np.linalg.norm(point1 - point2)


def calculate_velocity(current_pos: np.ndarray, previous_pos: np.ndarray, fps: float = 30.0) -> float:
    """
    Calculate velocity between two positions

    Args:
        current_pos: Current position
        previous_pos: Previous position
        fps: Frames per second

    Returns:
        Velocity in units per second
    """
    if previous_pos is None:
        return 0.0

    distance = calculate_distance(current_pos, previous_pos)
    velocity = distance * fps

    return velocity


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to be scale and translation invariant

    Args:
        landmarks: Array of shape (33, 3) containing x, y, z coordinates

    Returns:
        Normalized landmarks
    """
    if landmarks.shape[0] == 0:
        return landmarks

    # Center at hip midpoint
    left_hip = landmarks[config.KEY_LANDMARKS['left_hip']]
    right_hip = landmarks[config.KEY_LANDMARKS['right_hip']]
    hip_center = (left_hip + right_hip) / 2

    centered = landmarks - hip_center

    # Scale based on torso height
    left_shoulder = landmarks[config.KEY_LANDMARKS['left_shoulder']]
    right_shoulder = landmarks[config.KEY_LANDMARKS['right_shoulder']]
    shoulder_center = (left_shoulder + right_shoulder) / 2

    torso_height = np.linalg.norm(shoulder_center - hip_center)

    if torso_height > 1e-6:
        normalized = centered / torso_height
    else:
        normalized = centered

    return normalized


def smooth_predictions(predictions: List[int], window_size: int = 5) -> int:
    """
    Smooth predictions using a sliding window (majority vote)

    Args:
        predictions: List of recent predictions
        window_size: Size of smoothing window

    Returns:
        Smoothed prediction
    """
    if len(predictions) < window_size:
        window_size = len(predictions)

    recent_predictions = predictions[-window_size:]

    # Majority vote
    prediction_counts = {}
    for pred in recent_predictions:
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1

    smoothed_pred = max(prediction_counts, key=prediction_counts.get)

    return smoothed_pred


def draw_landmarks(image: np.ndarray, landmarks, connections: Optional[List] = None) -> np.ndarray:
    """
    Draw pose landmarks on image

    Args:
        image: Input image
        landmarks: MediaPipe landmarks
        connections: List of landmark connections to draw

    Returns:
        Image with landmarks drawn
    """
    if landmarks is None:
        return image

    h, w, _ = image.shape

    # Draw connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
                start = landmarks.landmark[start_idx]
                end = landmarks.landmark[end_idx]

                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(image, start_point, end_point, (255, 255, 255), 2)

    # Draw landmarks
    for idx, landmark in enumerate(landmarks.landmark):
        if landmark.visibility > 0.5:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

    return image


def put_text_with_background(image: np.ndarray, text: str, position: Tuple[int, int],
                              font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
                              thickness: int = 2) -> np.ndarray:
    """
    Draw text with a background rectangle for better visibility

    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness

    Returns:
        Image with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position

    # Draw background rectangle
    cv2.rectangle(image,
                  (x - 5, y - text_height - 5),
                  (x + text_width + 5, y + baseline + 5),
                  (0, 0, 0),
                  -1)

    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

    return image


def create_sequence_windows(features: np.ndarray, sequence_length: int,
                            overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from feature sequence

    Args:
        features: Array of shape (num_frames, feature_dim)
        sequence_length: Length of each window
        overlap: Overlap between windows (0 to 1)

    Returns:
        windows: Array of shape (num_windows, sequence_length, feature_dim)
        indices: Start indices of each window
    """
    num_frames = features.shape[0]
    step = int(sequence_length * (1 - overlap))

    if step < 1:
        step = 1

    windows = []
    indices = []

    for start in range(0, num_frames - sequence_length + 1, step):
        end = start + sequence_length
        windows.append(features[start:end])
        indices.append(start)

    if len(windows) == 0:
        # If sequence is too short, pad it
        if num_frames < sequence_length:
            padded = np.zeros((sequence_length, features.shape[1]))
            padded[:num_frames] = features
            windows.append(padded)
            indices.append(0)

    return np.array(windows), np.array(indices)


def augment_sequence(sequence: np.ndarray, config_aug: dict) -> np.ndarray:
    """
    Apply data augmentation to a sequence

    Args:
        sequence: Input sequence of shape (sequence_length, feature_dim)
        config_aug: Augmentation configuration

    Returns:
        Augmented sequence
    """
    augmented = sequence.copy()

    # Add noise
    if config_aug.get('noise_factor', 0) > 0:
        noise = np.random.normal(0, config_aug['noise_factor'], augmented.shape)
        augmented += noise

    # Scale
    if config_aug.get('scale_range', 0) > 0:
        scale = 1 + np.random.uniform(-config_aug['scale_range'],
                                      config_aug['scale_range'])
        augmented *= scale

    # Horizontal flip (only for angle and position features, not velocities)
    if config_aug.get('horizontal_flip', False) and np.random.rand() > 0.5:
        # This is simplified - in practice, you'd need to flip specific features
        pass

    return augmented


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """
    Calculate evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                average='weighted',
                                                                zero_division=0)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(num_classes)
    )

    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': conf_matrix
    }

    return metrics


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    import random
    random.seed(seed)
