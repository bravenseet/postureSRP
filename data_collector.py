"""
Data Collector for Compensatory Movement Detection
Processes batches of videos to extract features and prepare training data

Expected directory structure:
data/raw_videos/
├── participant_01/
│   ├── correct_form/
│   │   ├── video_001.mp4
│   │   └── video_002.mp4
│   ├── scapular_winging/
│   │   └── video_001.mp4
│   └── ...
├── participant_02/
│   └── ...
"""
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import config
from feature_extractor import FeatureExtractor
from utils import create_sequence_windows


class DataCollector:
    """
    Collects and processes video data for training
    """

    def __init__(self, video_dir: str = None):
        """
        Initialize data collector

        Args:
            video_dir: Directory containing participant videos
        """
        self.video_dir = video_dir or config.RAW_VIDEO_DIR
        self.feature_extractor = FeatureExtractor()

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )

        self.data = []
        self.metadata = []

    def collect_all_data(self, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all videos in the video directory

        Args:
            save_path: Path to save processed data

        Returns:
            X: Feature sequences of shape (num_samples, sequence_length, feature_dim)
            y: Labels of shape (num_samples,)
        """
        if not os.path.exists(self.video_dir):
            raise ValueError(f"Video directory not found: {self.video_dir}")

        print(f"Collecting data from: {self.video_dir}")
        print(f"Looking for {config.NUM_CLASSES} movement classes")

        # Iterate through participants
        participants = [d for d in os.listdir(self.video_dir)
                       if os.path.isdir(os.path.join(self.video_dir, d))]

        if len(participants) == 0:
            raise ValueError(f"No participant directories found in {self.video_dir}")

        print(f"Found {len(participants)} participants")

        all_sequences = []
        all_labels = []

        for participant_id in tqdm(participants, desc="Processing participants"):
            participant_path = os.path.join(self.video_dir, participant_id)

            # Iterate through movement classes
            for class_id, class_name in config.MOVEMENT_CLASSES.items():
                class_path = os.path.join(participant_path, class_name)

                if not os.path.exists(class_path):
                    print(f"Warning: Class directory not found: {class_path}")
                    continue

                # Process all videos in this class
                video_files = [f for f in os.listdir(class_path)
                             if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]

                for video_file in video_files:
                    video_path = os.path.join(class_path, video_file)

                    try:
                        sequences, labels = self.process_video(
                            video_path,
                            class_id,
                            participant_id,
                            video_file
                        )

                        if len(sequences) > 0:
                            all_sequences.extend(sequences)
                            all_labels.extend(labels)
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
                        continue

        if len(all_sequences) == 0:
            raise ValueError("No data collected. Please check your video directory structure.")

        X = np.array(all_sequences, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)

        # Ensure y is 1D (sparse_categorical_crossentropy expects this)
        if len(y.shape) > 1:
            y = y.flatten()

        print(f"\nData collection complete!")
        print(f"Total sequences: {len(X)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Label distribution:")
        for class_id, class_name in config.MOVEMENT_CLASSES.items():
            count = np.sum(y == class_id)
            print(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)")

        # Save processed data
        if save_path is None:
            save_path = os.path.join(config.PROCESSED_DATA_DIR, 'training_data.npz')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, X=X, y=y, metadata=self.metadata)
        print(f"\nData saved to: {save_path}")

        # Save metadata separately
        metadata_path = save_path.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        return X, y

    def process_video(self, video_path: str, label: int,
                     participant_id: str = None, video_id: str = None) -> Tuple[List, List]:
        """
        Process a single video file

        Args:
            video_path: Path to video file
            label: Class label
            participant_id: Participant identifier
            video_id: Video identifier

        Returns:
            sequences: List of feature sequences
            labels: List of labels (one per sequence)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Reset feature extractor
        self.feature_extractor.reset()
        self.feature_extractor.fps = fps

        frame_features = []
        frame_count = 0
        processed_count = 0

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Skip frames if configured
            if frame_count % config.VIDEO_PROCESSING['skip_frames'] != 0:
                frame_count += 1
                continue

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract features
                features = self.feature_extractor.extract_features(results.pose_landmarks)

                # Validate feature dimensions
                if len(features) >= config.MIN_FEATURE_DIM:
                    frame_features.append(features)
                    processed_count += 1
                else:
                    print(f"Warning: Frame {frame_count} has only {len(features)} features (min: {config.MIN_FEATURE_DIM})")

            frame_count += 1

        cap.release()

        if len(frame_features) == 0:
            print(f"Warning: No valid pose features extracted from {video_path}")
            return [], []

        print(f"Processed {processed_count}/{frame_count} frames with valid features")

        # Convert to numpy array
        frame_features = np.array(frame_features)

        # Create sequences using sliding window
        sequences, _ = create_sequence_windows(
            frame_features,
            config.SEQUENCE_LENGTH,
            overlap=0.5
        )

        # Create labels for each sequence
        labels = [label] * len(sequences)

        # Store metadata
        for i in range(len(sequences)):
            self.metadata.append({
                'participant_id': participant_id,
                'video_id': video_id,
                'video_path': video_path,
                'label': int(label),
                'class_name': config.MOVEMENT_CLASSES[label],
                'sequence_index': i,
                'total_frames': processed_count,
                'fps': fps
            })

        return sequences.tolist(), labels

    def process_single_video_for_reps(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Process video and segment by repetitions

        Args:
            video_path: Path to video file

        Returns:
            rep_sequences: List of sequences, one per repetition
            rep_labels: Corresponding labels
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.feature_extractor.reset()
        self.feature_extractor.fps = fps

        frame_features = []
        elbow_angles = []

        # Extract features from all frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                features = self.feature_extractor.extract_features(results.pose_landmarks)

                # Validate feature dimensions
                if len(features) >= config.MIN_FEATURE_DIM:
                    frame_features.append(features)

                    # Store elbow angle for rep detection (average of left and right)
                    elbow_angle = (features[0] + features[1]) / 2
                    elbow_angles.append(elbow_angle)

        cap.release()

        if len(frame_features) == 0:
            return [], []

        # Detect repetitions based on elbow angle
        reps = self._detect_reps(elbow_angles)

        # Create sequences for each rep
        rep_sequences = []
        for start_idx, end_idx in reps:
            rep_features = frame_features[start_idx:end_idx]

            if len(rep_features) >= config.SEQUENCE_LENGTH:
                # Resample to fixed length
                rep_features = self._resample_sequence(rep_features, config.SEQUENCE_LENGTH)
                rep_sequences.append(np.array(rep_features))
            elif len(rep_features) >= config.VIDEO_PROCESSING['min_rep_duration']:
                # Pad short sequences
                padded = np.zeros((config.SEQUENCE_LENGTH, config.FEATURE_DIM))
                padded[:len(rep_features)] = rep_features
                rep_sequences.append(padded)

        return rep_sequences, []

    def _detect_reps(self, elbow_angles: List[float]) -> List[Tuple[int, int]]:
        """
        Detect pushup repetitions based on elbow angle

        Args:
            elbow_angles: List of elbow angles over time

        Returns:
            List of (start_idx, end_idx) tuples for each rep
        """
        reps = []
        state = 'up'  # 'up' or 'down'
        rep_start = 0

        threshold_down = config.REP_COUNTER['elbow_angle_threshold_down']
        threshold_up = config.REP_COUNTER['elbow_angle_threshold_up']
        min_frames = config.REP_COUNTER['min_frames_between_reps']

        for i, angle in enumerate(elbow_angles):
            if state == 'up' and angle < threshold_down:
                # Transition to down position
                state = 'down'

            elif state == 'down' and angle > threshold_up:
                # Transition back to up position - rep complete
                if i - rep_start >= min_frames:
                    reps.append((rep_start, i))
                rep_start = i
                state = 'up'

        return reps

    def _resample_sequence(self, sequence: List[np.ndarray], target_length: int) -> np.ndarray:
        """
        Resample sequence to target length

        Args:
            sequence: Original sequence
            target_length: Desired length

        Returns:
            Resampled sequence
        """
        sequence = np.array(sequence)
        original_length = len(sequence)

        if original_length == target_length:
            return sequence

        # Linear interpolation
        indices = np.linspace(0, original_length - 1, target_length)
        resampled = np.zeros((target_length, sequence.shape[1]))

        for i, idx in enumerate(indices):
            lower_idx = int(np.floor(idx))
            upper_idx = int(np.ceil(idx))

            if lower_idx == upper_idx:
                resampled[i] = sequence[lower_idx]
            else:
                # Linear interpolation
                weight = idx - lower_idx
                resampled[i] = (1 - weight) * sequence[lower_idx] + weight * sequence[upper_idx]

        return resampled

    def visualize_sample(self, video_path: str, output_path: str = None):
        """
        Visualize pose detection on a sample video

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Display or save
            if writer:
                writer.write(frame)
            else:
                cv2.imshow('Pose Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
            print(f"Visualization saved to: {output_path}")
        else:
            cv2.destroyAllWindows()

    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the collected dataset

        Returns:
            Dictionary containing dataset statistics
        """
        if len(self.metadata) == 0:
            return {}

        df = pd.DataFrame(self.metadata)

        stats = {
            'total_sequences': len(df),
            'total_participants': df['participant_id'].nunique(),
            'total_videos': df['video_id'].nunique(),
            'class_distribution': df['class_name'].value_counts().to_dict(),
            'sequences_per_participant': df.groupby('participant_id').size().to_dict(),
            'sequences_per_video': df.groupby('video_id').size().mean(),
            'avg_fps': df['fps'].mean()
        }

        return stats


def main():
    """
    Example usage of DataCollector
    """
    import argparse

    parser = argparse.ArgumentParser(description='Collect training data from videos')
    parser.add_argument('--video_dir', type=str, default=None,
                       help='Directory containing participant videos')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for processed data')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Visualize pose detection on a sample video')

    args = parser.parse_args()

    collector = DataCollector(video_dir=args.video_dir)

    if args.visualize:
        print(f"Visualizing: {args.visualize}")
        output_viz = args.visualize.replace('.mp4', '_visualized.mp4')
        collector.visualize_sample(args.visualize, output_viz)
    else:
        X, y = collector.collect_all_data(save_path=args.output)

        # Print statistics
        stats = collector.get_dataset_statistics()
        print("\nDataset Statistics:")
        print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()