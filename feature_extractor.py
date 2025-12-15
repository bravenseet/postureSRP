"""
Feature Extractor for Compensatory Movement Detection
Extracts 78 biomechanical features from pose landmarks

Feature Categories:
1. Joint Angles (24 features)
2. Body Alignment (12 features)
3. Velocities (15 features)
4. Distances and Ratios (12 features)
5. Temporal Features (9 features)
6. Stability Metrics (6 features)
"""
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import config
from utils import calculate_angle, calculate_distance, calculate_velocity, normalize_landmarks


class FeatureExtractor:
    """
    Extracts biomechanical features from MediaPipe pose landmarks
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.previous_landmarks = None
        self.previous_features = None
        self.frame_count = 0
        self.fps = 30.0

    def extract_features(self, landmarks, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Extract 78 biomechanical features from pose landmarks

        Args:
            landmarks: MediaPipe pose landmarks
            timestamp: Optional timestamp for velocity calculation

        Returns:
            Feature vector of shape (78,)
        """
        if landmarks is None:
            return np.zeros(config.FEATURE_DIM)

        # Convert landmarks to numpy array
        landmark_array = self._landmarks_to_array(landmarks)

        # Normalize landmarks for scale invariance
        normalized_landmarks = normalize_landmarks(landmark_array)

        # Extract feature categories
        features = []

        # 1. Joint Angles (24 features)
        angle_features = self._extract_angle_features(normalized_landmarks, landmark_array)
        features.extend(angle_features)

        # 2. Body Alignment (12 features)
        alignment_features = self._extract_alignment_features(normalized_landmarks, landmark_array)
        features.extend(alignment_features)

        # 3. Velocities (15 features)
        velocity_features = self._extract_velocity_features(landmark_array)
        features.extend(velocity_features)

        # 4. Distances and Ratios (12 features)
        distance_features = self._extract_distance_features(normalized_landmarks, landmark_array)
        features.extend(distance_features)

        # 5. Temporal Features (9 features)
        temporal_features = self._extract_temporal_features(angle_features)
        features.extend(temporal_features)

        # 6. Stability Metrics (6 features)
        stability_features = self._extract_stability_features(normalized_landmarks)
        features.extend(stability_features)

        # Store for next frame
        self.previous_landmarks = landmark_array
        self.previous_features = np.array(features)
        self.frame_count += 1

        feature_vector = np.array(features, dtype=np.float32)

        # Ensure correct dimension
        if len(feature_vector) != config.FEATURE_DIM:
            print(f"Warning: Feature vector has {len(feature_vector)} features, expected {config.FEATURE_DIM}")
            # Pad or truncate if necessary
            if len(feature_vector) < config.FEATURE_DIM:
                feature_vector = np.pad(feature_vector, (0, config.FEATURE_DIM - len(feature_vector)))
            else:
                feature_vector = feature_vector[:config.FEATURE_DIM]

        return feature_vector

    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """
        Convert MediaPipe landmarks to numpy array

        Returns:
            Array of shape (33, 3) containing x, y, z coordinates
        """
        landmark_array = np.zeros((33, 3))

        for i, landmark in enumerate(landmarks.landmark):
            landmark_array[i] = [landmark.x, landmark.y, landmark.z]

        return landmark_array

    def _extract_angle_features(self, normalized_landmarks: np.ndarray,
                                raw_landmarks: np.ndarray) -> List[float]:
        """
        Extract 24 joint angle features
        """
        features = []
        lm = config.KEY_LANDMARKS

        # Elbow angles (2)
        left_elbow_angle = calculate_angle(
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['left_elbow']],
            raw_landmarks[lm['left_wrist']]
        )
        right_elbow_angle = calculate_angle(
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['right_elbow']],
            raw_landmarks[lm['right_wrist']]
        )
        features.extend([left_elbow_angle, right_elbow_angle])

        # Shoulder angles (2)
        left_shoulder_angle = calculate_angle(
            raw_landmarks[lm['left_hip']],
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['left_elbow']]
        )
        right_shoulder_angle = calculate_angle(
            raw_landmarks[lm['right_hip']],
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['right_elbow']]
        )
        features.extend([left_shoulder_angle, right_shoulder_angle])

        # Hip angles (2)
        left_hip_angle = calculate_angle(
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['left_hip']],
            raw_landmarks[lm['left_knee']]
        )
        right_hip_angle = calculate_angle(
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['right_hip']],
            raw_landmarks[lm['right_knee']]
        )
        features.extend([left_hip_angle, right_hip_angle])

        # Knee angles (2)
        left_knee_angle = calculate_angle(
            raw_landmarks[lm['left_hip']],
            raw_landmarks[lm['left_knee']],
            raw_landmarks[lm['left_ankle']]
        )
        right_knee_angle = calculate_angle(
            raw_landmarks[lm['right_hip']],
            raw_landmarks[lm['right_knee']],
            raw_landmarks[lm['right_ankle']]
        )
        features.extend([left_knee_angle, right_knee_angle])

        # Trunk angle (angle between shoulder-hip line and vertical) (2)
        left_trunk_angle = self._calculate_trunk_angle(
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['left_hip']]
        )
        right_trunk_angle = self._calculate_trunk_angle(
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['right_hip']]
        )
        features.extend([left_trunk_angle, right_trunk_angle])

        # Elbow flare angles (angle between elbow-shoulder and torso) (2)
        left_elbow_flare = self._calculate_elbow_flare(
            raw_landmarks[lm['left_elbow']],
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['left_hip']]
        )
        right_elbow_flare = self._calculate_elbow_flare(
            raw_landmarks[lm['right_elbow']],
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['right_hip']]
        )
        features.extend([left_elbow_flare, right_elbow_flare])

        # Scapular angles (shoulder protraction) (2)
        left_scapular_angle = calculate_angle(
            raw_landmarks[lm['left_elbow']],
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['right_shoulder']]
        )
        right_scapular_angle = calculate_angle(
            raw_landmarks[lm['right_elbow']],
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['left_shoulder']]
        )
        features.extend([left_scapular_angle, right_scapular_angle])

        # Ankle angles (2)
        left_ankle_angle = calculate_angle(
            raw_landmarks[lm['left_knee']],
            raw_landmarks[lm['left_ankle']],
            raw_landmarks[lm['left_foot_index']]
        )
        right_ankle_angle = calculate_angle(
            raw_landmarks[lm['right_knee']],
            raw_landmarks[lm['right_ankle']],
            raw_landmarks[lm['right_foot_index']]
        )
        features.extend([left_ankle_angle, right_ankle_angle])

        # Full body angle (shoulder-hip-ankle) (2)
        left_body_angle = calculate_angle(
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['left_hip']],
            raw_landmarks[lm['left_ankle']]
        )
        right_body_angle = calculate_angle(
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['right_hip']],
            raw_landmarks[lm['right_ankle']]
        )
        features.extend([left_body_angle, right_body_angle])

        # Wrist angles (2)
        left_wrist_angle = calculate_angle(
            raw_landmarks[lm['left_elbow']],
            raw_landmarks[lm['left_wrist']],
            raw_landmarks[lm['left_index']]
        )
        right_wrist_angle = calculate_angle(
            raw_landmarks[lm['right_elbow']],
            raw_landmarks[lm['right_wrist']],
            raw_landmarks[lm['right_index']]
        )
        features.extend([left_wrist_angle, right_wrist_angle])

        # Head-neck-torso angle (2)
        head_torso_angle_left = calculate_angle(
            raw_landmarks[lm['nose']],
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['left_hip']]
        )
        head_torso_angle_right = calculate_angle(
            raw_landmarks[lm['nose']],
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['right_hip']]
        )
        features.extend([head_torso_angle_left, head_torso_angle_right])

        # Total: 24 angle features
        return features

    def _extract_alignment_features(self, normalized_landmarks: np.ndarray,
                                   raw_landmarks: np.ndarray) -> List[float]:
        """
        Extract 12 body alignment features
        """
        features = []
        lm = config.KEY_LANDMARKS

        # Shoulder alignment (horizontal difference) (1)
        shoulder_y_diff = abs(raw_landmarks[lm['left_shoulder']][1] -
                             raw_landmarks[lm['right_shoulder']][1])
        features.append(shoulder_y_diff)

        # Hip alignment (1)
        hip_y_diff = abs(raw_landmarks[lm['left_hip']][1] -
                        raw_landmarks[lm['right_hip']][1])
        features.append(hip_y_diff)

        # Shoulder-hip vertical alignment (should be aligned in plank) (2)
        left_shoulder_hip_x_diff = abs(raw_landmarks[lm['left_shoulder']][0] -
                                      raw_landmarks[lm['left_hip']][0])
        right_shoulder_hip_x_diff = abs(raw_landmarks[lm['right_shoulder']][0] -
                                       raw_landmarks[lm['right_hip']][0])
        features.extend([left_shoulder_hip_x_diff, right_shoulder_hip_x_diff])

        # Hip height (relative to shoulders and ankles) (1)
        shoulder_center_y = (raw_landmarks[lm['left_shoulder']][1] +
                            raw_landmarks[lm['right_shoulder']][1]) / 2
        hip_center_y = (raw_landmarks[lm['left_hip']][1] +
                       raw_landmarks[lm['right_hip']][1]) / 2
        ankle_center_y = (raw_landmarks[lm['left_ankle']][1] +
                         raw_landmarks[lm['right_ankle']][1]) / 2

        hip_height_ratio = (hip_center_y - shoulder_center_y) / (ankle_center_y - shoulder_center_y + 1e-6)
        features.append(hip_height_ratio)

        # Lateral deviation (distance from center line) (2)
        center_x = (raw_landmarks[lm['left_hip']][0] + raw_landmarks[lm['right_hip']][0]) / 2

        left_shoulder_deviation = abs(raw_landmarks[lm['left_shoulder']][0] - center_x)
        right_shoulder_deviation = abs(raw_landmarks[lm['right_shoulder']][0] - center_x)
        features.extend([left_shoulder_deviation, right_shoulder_deviation])

        # Elbow width (distance between elbows) (1)
        elbow_width = calculate_distance(
            raw_landmarks[lm['left_elbow']],
            raw_landmarks[lm['right_elbow']]
        )
        features.append(elbow_width)

        # Hand width (distance between wrists) (1)
        hand_width = calculate_distance(
            raw_landmarks[lm['left_wrist']],
            raw_landmarks[lm['right_wrist']]
        )
        features.append(hand_width)

        # Torso rotation (z-axis difference between shoulders) (1)
        shoulder_z_diff = abs(raw_landmarks[lm['left_shoulder']][2] -
                             raw_landmarks[lm['right_shoulder']][2])
        features.append(shoulder_z_diff)

        # Planarity (how flat the body is - variance in z-coordinates) (1)
        key_points = [lm['left_shoulder'], lm['right_shoulder'],
                     lm['left_hip'], lm['right_hip'],
                     lm['left_ankle'], lm['right_ankle']]
        z_coords = [raw_landmarks[i][2] for i in key_points]
        z_variance = np.var(z_coords)
        features.append(z_variance)

        # Total: 12 alignment features
        return features

    def _extract_velocity_features(self, landmarks: np.ndarray) -> List[float]:
        """
        Extract 15 velocity features
        """
        features = []
        lm = config.KEY_LANDMARKS

        if self.previous_landmarks is None:
            return [0.0] * 15

        # Joint velocities (9)
        key_joints = [
            'left_wrist', 'right_wrist',
            'left_elbow', 'right_elbow',
            'left_shoulder', 'right_shoulder',
            'left_hip', 'right_hip',
            'nose'
        ]

        for joint in key_joints:
            velocity = calculate_velocity(
                landmarks[lm[joint]],
                self.previous_landmarks[lm[joint]],
                self.fps
            )
            features.append(velocity)

        # Center of mass velocity (3 components: x, y, z)
        com_current = np.mean(landmarks, axis=0)
        com_previous = np.mean(self.previous_landmarks, axis=0)
        com_velocity = (com_current - com_previous) * self.fps
        features.extend(com_velocity.tolist())

        # Angular velocity of elbow (rate of angle change) (2)
        if self.previous_features is not None and len(self.previous_features) >= 2:
            left_elbow_angular_vel = (self.previous_features[0] - features[0]) * self.fps
            right_elbow_angular_vel = (self.previous_features[1] - features[1]) * self.fps
            features.extend([left_elbow_angular_vel, right_elbow_angular_vel])
        else:
            features.extend([0.0, 0.0])

        # Hip vertical velocity (1)
        hip_center_current = (landmarks[lm['left_hip']] + landmarks[lm['right_hip']]) / 2
        hip_center_previous = (self.previous_landmarks[lm['left_hip']] +
                              self.previous_landmarks[lm['right_hip']]) / 2
        hip_vertical_velocity = (hip_center_current[1] - hip_center_previous[1]) * self.fps
        features.append(hip_vertical_velocity)

        # Total: 15 velocity features
        return features

    def _extract_distance_features(self, normalized_landmarks: np.ndarray,
                                   raw_landmarks: np.ndarray) -> List[float]:
        """
        Extract 12 distance and ratio features
        """
        features = []
        lm = config.KEY_LANDMARKS

        # Wrist-ankle distance (2)
        left_wrist_ankle = calculate_distance(
            raw_landmarks[lm['left_wrist']],
            raw_landmarks[lm['left_ankle']]
        )
        right_wrist_ankle = calculate_distance(
            raw_landmarks[lm['right_wrist']],
            raw_landmarks[lm['right_ankle']]
        )
        features.extend([left_wrist_ankle, right_wrist_ankle])

        # Shoulder-wrist distance (2)
        left_shoulder_wrist = calculate_distance(
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['left_wrist']]
        )
        right_shoulder_wrist = calculate_distance(
            raw_landmarks[lm['right_shoulder']],
            raw_landmarks[lm['right_wrist']]
        )
        features.extend([left_shoulder_wrist, right_shoulder_wrist])

        # Hip-shoulder distance (2)
        left_hip_shoulder = calculate_distance(
            raw_landmarks[lm['left_hip']],
            raw_landmarks[lm['left_shoulder']]
        )
        right_hip_shoulder = calculate_distance(
            raw_landmarks[lm['right_hip']],
            raw_landmarks[lm['right_shoulder']]
        )
        features.extend([left_hip_shoulder, right_hip_shoulder])

        # Shoulder width (1)
        shoulder_width = calculate_distance(
            raw_landmarks[lm['left_shoulder']],
            raw_landmarks[lm['right_shoulder']]
        )
        features.append(shoulder_width)

        # Hip width (1)
        hip_width = calculate_distance(
            raw_landmarks[lm['left_hip']],
            raw_landmarks[lm['right_hip']]
        )
        features.append(hip_width)

        # Torso length (1)
        torso_length = (left_hip_shoulder + right_hip_shoulder) / 2
        features.append(torso_length)

        # Aspect ratio (width/height) (1)
        body_height = raw_landmarks[lm['nose']][1] - (raw_landmarks[lm['left_ankle']][1] +
                                                      raw_landmarks[lm['right_ankle']][1]) / 2
        aspect_ratio = shoulder_width / (abs(body_height) + 1e-6)
        features.append(aspect_ratio)

        # Hand-shoulder width ratio (1)
        hand_width = calculate_distance(raw_landmarks[lm['left_wrist']],
                                       raw_landmarks[lm['right_wrist']])
        hand_shoulder_ratio = hand_width / (shoulder_width + 1e-6)
        features.append(hand_shoulder_ratio)

        # Elbow-shoulder distance (1)
        elbow_shoulder_dist = (
            calculate_distance(raw_landmarks[lm['left_elbow']], raw_landmarks[lm['left_shoulder']]) +
            calculate_distance(raw_landmarks[lm['right_elbow']], raw_landmarks[lm['right_shoulder']])
        ) / 2
        features.append(elbow_shoulder_dist)

        # Total: 12 distance features
        return features

    def _extract_temporal_features(self, angle_features: List[float]) -> List[float]:
        """
        Extract 9 temporal features (changes over time)
        """
        features = []

        if self.previous_features is None or len(self.previous_features) < 24:
            return [0.0] * 9

        # Rate of change for key angles (6)
        # Elbows, shoulders, hips
        for i in [0, 1, 2, 3, 4, 5]:
            if i < len(angle_features) and i < len(self.previous_features):
                rate = angle_features[i] - self.previous_features[i]
                features.append(rate)
            else:
                features.append(0.0)

        # Acceleration (second derivative) for elbow angles (2)
        if len(self.previous_features) >= 24:
            left_elbow_accel = (angle_features[0] - 2 * self.previous_features[0] +
                               self.previous_features[0])
            right_elbow_accel = (angle_features[1] - 2 * self.previous_features[1] +
                                self.previous_features[1])
            features.extend([left_elbow_accel, right_elbow_accel])
        else:
            features.extend([0.0, 0.0])

        # Movement smoothness (jerk - third derivative) (1)
        if len(features) >= 8:
            jerk = np.std(features[:6])
            features.append(jerk)
        else:
            features.append(0.0)

        # Total: 9 temporal features
        return features

    def _extract_stability_features(self, normalized_landmarks: np.ndarray) -> List[float]:
        """
        Extract 6 stability metrics
        """
        features = []
        lm = config.KEY_LANDMARKS

        # Bilateral symmetry (difference between left and right) (3)
        # Shoulders
        shoulder_symmetry = abs(
            np.linalg.norm(normalized_landmarks[lm['left_shoulder']] -
                          normalized_landmarks[lm['left_hip']]) -
            np.linalg.norm(normalized_landmarks[lm['right_shoulder']] -
                          normalized_landmarks[lm['right_hip']])
        )
        features.append(shoulder_symmetry)

        # Elbows
        elbow_symmetry = abs(
            np.linalg.norm(normalized_landmarks[lm['left_elbow']] -
                          normalized_landmarks[lm['left_shoulder']]) -
            np.linalg.norm(normalized_landmarks[lm['right_elbow']] -
                          normalized_landmarks[lm['right_shoulder']])
        )
        features.append(elbow_symmetry)

        # Hips
        hip_symmetry = abs(
            np.linalg.norm(normalized_landmarks[lm['left_hip']] -
                          normalized_landmarks[lm['left_knee']]) -
            np.linalg.norm(normalized_landmarks[lm['right_hip']] -
                          normalized_landmarks[lm['right_knee']])
        )
        features.append(hip_symmetry)

        # Center of mass deviation from midline (1)
        midline_x = (normalized_landmarks[lm['left_hip']][0] +
                    normalized_landmarks[lm['right_hip']][0]) / 2
        com = np.mean(normalized_landmarks, axis=0)
        com_deviation = abs(com[0] - midline_x)
        features.append(com_deviation)

        # Postural sway (variance in key points) (1)
        key_points = [lm['left_shoulder'], lm['right_shoulder'],
                     lm['left_hip'], lm['right_hip']]
        y_coords = [normalized_landmarks[i][1] for i in key_points]
        postural_sway = np.var(y_coords)
        features.append(postural_sway)

        # Overall body stability (coefficient of variation) (1)
        if self.previous_landmarks is not None:
            movement = np.linalg.norm(normalized_landmarks - normalize_landmarks(self.previous_landmarks))
            features.append(movement)
        else:
            features.append(0.0)

        # Total: 6 stability features
        return features

    def _calculate_trunk_angle(self, shoulder: np.ndarray, hip: np.ndarray) -> float:
        """
        Calculate angle of trunk relative to vertical
        """
        vertical = np.array([0, 1, 0])
        trunk_vector = hip - shoulder

        norm = np.linalg.norm(trunk_vector)
        if norm < 1e-6:
            return 0.0

        trunk_normalized = trunk_vector / norm
        dot_product = np.dot(trunk_normalized, vertical)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        angle = np.degrees(np.arccos(abs(dot_product)))
        return angle

    def _calculate_elbow_flare(self, elbow: np.ndarray, shoulder: np.ndarray,
                               hip: np.ndarray) -> float:
        """
        Calculate elbow flare angle (angle between arm and torso)
        """
        torso_vector = hip - shoulder
        arm_vector = elbow - shoulder

        # Project arm vector onto horizontal plane perpendicular to torso
        torso_norm = np.linalg.norm(torso_vector)
        if torso_norm < 1e-6:
            return 0.0

        torso_normalized = torso_vector / torso_norm

        # Calculate angle in horizontal plane
        arm_horizontal = arm_vector - np.dot(arm_vector, torso_normalized) * torso_normalized
        arm_h_norm = np.linalg.norm(arm_horizontal)

        if arm_h_norm < 1e-6:
            return 0.0

        # Angle from body midline
        angle = np.degrees(np.arctan2(arm_h_norm, np.dot(arm_vector, torso_normalized)))

        return angle

    def reset(self):
        """Reset the feature extractor state"""
        self.previous_landmarks = None
        self.previous_features = None
        self.frame_count = 0
