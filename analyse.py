"""
Analysis Module for Compensatory Movement Detection
Analyzes exercise performance without visual feedback (for research comparison)
"""
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow import keras
from collections import deque
import json
import os
from datetime import datetime
import config
from feature_extractor import FeatureExtractor
from utils import smooth_predictions


class PerformanceAnalyzer:
    """
    Analyzes exercise performance and generates detailed reports
    """

    def __init__(self, model_path: str):
        """
        Initialize analyzer

        Args:
            model_path: Path to trained model
        """
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Buffers
        self.feature_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=config.INFERENCE_CONFIG['smoothing_window'])

    def analyze_video(self, video_path: str, participant_id: str = None,
                     save_results: bool = True) -> dict:
        """
        Analyze a video without providing real-time feedback

        Args:
            video_path: Path to video file
            participant_id: Participant identifier
            save_results: Whether to save analysis results

        Returns:
            Analysis results dictionary
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"Analyzing video: {video_path}")
        print(f"Duration: {duration:.1f}s, FPS: {fps:.1f}, Frames: {total_frames}")

        # Reset state
        self.reset()
        self.feature_extractor.fps = fps

        # Storage for analysis
        frame_data = []
        predictions = []
        confidences = []
        rep_indices = []

        frame_count = 0
        rep_count = 0
        rep_state = 'up'

        # Process video
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract features
                features = self.feature_extractor.extract_features(results.pose_landmarks)

                # Validate feature dimensions
                if len(features) >= config.MIN_FEATURE_DIM:
                    self.feature_buffer.append(features)
                else:
                    # Skip frames with too few features
                    frame_count += 1
                    continue

                # Make prediction if buffer is full
                if len(self.feature_buffer) == config.SEQUENCE_LENGTH:
                    try:
                        sequence = np.array(list(self.feature_buffer))

                        # Validate and fix sequence dimensions
                        if sequence.shape[1] != config.FEATURE_DIM:
                            if sequence.shape[1] < config.FEATURE_DIM:
                                padding = np.full((sequence.shape[0], config.FEATURE_DIM - sequence.shape[1]),
                                                 config.FEATURE_PADDING_VALUE)
                                sequence = np.concatenate([sequence, padding], axis=1)
                            else:
                                sequence = sequence[:, :config.FEATURE_DIM]

                        # Clean NaN/Inf values
                        sequence = np.nan_to_num(sequence, nan=config.FEATURE_PADDING_VALUE,
                                                posinf=1e6, neginf=-1e6)
                        sequence = sequence[np.newaxis, :, :]

                        # Predict
                        pred_probs = self.model.predict(sequence, verbose=0)
                        confidence = np.max(pred_probs[0])
                        prediction = np.argmax(pred_probs[0])
                    except Exception as e:
                        print(f"Error making prediction at frame {frame_count}: {e}")
                        frame_count += 1
                        continue

                    # Smooth prediction
                    self.prediction_buffer.append(prediction)
                    smoothed_pred = smooth_predictions(
                        list(self.prediction_buffer),
                        config.INFERENCE_CONFIG['smoothing_window']
                    )

                    predictions.append(smoothed_pred)
                    confidences.append(confidence)

                    # Detect reps
                    avg_elbow_angle = (features[0] + features[1]) / 2

                    if rep_state == 'up' and avg_elbow_angle < config.REP_COUNTER['elbow_angle_threshold_down']:
                        rep_state = 'down'
                    elif rep_state == 'down' and avg_elbow_angle > config.REP_COUNTER['elbow_angle_threshold_up']:
                        rep_count += 1
                        rep_indices.append(frame_count)
                        rep_state = 'up'

                    # Store frame data
                    frame_data.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'prediction': int(smoothed_pred),
                        'class_name': config.MOVEMENT_CLASSES[smoothed_pred],
                        'confidence': float(confidence),
                        'elbow_angle': float(avg_elbow_angle),
                        'is_rep_complete': frame_count in rep_indices
                    })

            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()

        print(f"Analysis complete. Detected {rep_count} reps")

        # Generate analysis report
        analysis_results = self._generate_report(
            frame_data,
            predictions,
            confidences,
            rep_indices,
            video_path,
            participant_id,
            fps,
            duration
        )

        # Save results if requested
        if save_results:
            self._save_results(analysis_results, video_path, participant_id)

        return analysis_results

    def _generate_report(self, frame_data: list, predictions: list,
                        confidences: list, rep_indices: list,
                        video_path: str, participant_id: str,
                        fps: float, duration: float) -> dict:
        """
        Generate comprehensive analysis report

        Args:
            frame_data: Per-frame data
            predictions: List of predictions
            confidences: List of confidence scores
            rep_indices: Frame indices where reps were completed
            video_path: Path to video
            participant_id: Participant ID
            fps: Video FPS
            duration: Video duration

        Returns:
            Analysis report dictionary
        """
        if len(predictions) == 0:
            return {
                'error': 'No predictions made - insufficient data',
                'video_path': video_path,
                'participant_id': participant_id
            }

        predictions = np.array(predictions)
        confidences = np.array(confidences)

        # Overall statistics
        total_reps = len(rep_indices)
        avg_confidence = np.mean(confidences)

        # Class distribution
        class_distribution = {}
        class_percentage = {}
        for class_id, class_name in config.MOVEMENT_CLASSES.items():
            count = np.sum(predictions == class_id)
            class_distribution[class_name] = int(count)
            class_percentage[class_name] = float(count / len(predictions) * 100)

        # Dominant compensation pattern
        if total_reps > 0:
            most_common_class = np.argmax([class_distribution[name] for name in config.MOVEMENT_CLASSES.values()])
            dominant_pattern = config.MOVEMENT_CLASSES[most_common_class]
        else:
            dominant_pattern = None

        # Form quality score (percentage of correct form)
        correct_form_percentage = class_percentage.get('correct_form', 0.0)

        # Temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(frame_data, rep_indices)

        # Generate recommendations
        recommendations = self._generate_recommendations(class_distribution, class_percentage)

        # Compile report
        report = {
            'metadata': {
                'video_path': video_path,
                'participant_id': participant_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'video_duration': float(duration),
                'fps': float(fps),
                'total_frames_analyzed': len(predictions)
            },
            'performance_summary': {
                'total_reps': total_reps,
                'avg_confidence': float(avg_confidence),
                'form_quality_score': float(correct_form_percentage),
                'dominant_pattern': dominant_pattern
            },
            'class_distribution': {
                'counts': class_distribution,
                'percentages': class_percentage
            },
            'temporal_analysis': temporal_analysis,
            'recommendations': recommendations,
            'detailed_frame_data': frame_data
        }

        return report

    def _analyze_temporal_patterns(self, frame_data: list, rep_indices: list) -> dict:
        """
        Analyze temporal patterns in the data

        Args:
            frame_data: Per-frame data
            rep_indices: Frame indices of completed reps

        Returns:
            Temporal analysis dictionary
        """
        if len(frame_data) == 0 or len(rep_indices) == 0:
            return {
                'reps_per_minute': 0,
                'avg_rep_duration': 0,
                'consistency_score': 0
            }

        df = pd.DataFrame(frame_data)

        # Calculate reps per minute
        duration_minutes = (df['timestamp'].max() - df['timestamp'].min()) / 60
        reps_per_minute = len(rep_indices) / duration_minutes if duration_minutes > 0 else 0

        # Calculate average rep duration
        if len(rep_indices) > 1:
            rep_durations = np.diff(rep_indices) / df['timestamp'].iloc[0]  # Convert to seconds
            avg_rep_duration = np.mean(rep_durations)
            rep_duration_std = np.std(rep_durations)
            consistency_score = 1 / (1 + rep_duration_std) if rep_duration_std > 0 else 1.0
        else:
            avg_rep_duration = 0
            consistency_score = 0

        # Analyze form changes over time
        form_changes = []
        for i in range(1, len(df)):
            if df.iloc[i]['prediction'] != df.iloc[i-1]['prediction']:
                form_changes.append(df.iloc[i]['timestamp'])

        return {
            'reps_per_minute': float(reps_per_minute),
            'avg_rep_duration': float(avg_rep_duration),
            'consistency_score': float(consistency_score),
            'form_transitions': len(form_changes),
            'avg_time_between_transitions': float(np.mean(np.diff(form_changes))) if len(form_changes) > 1 else 0
        }

    def _generate_recommendations(self, class_distribution: dict,
                                  class_percentage: dict) -> list:
        """
        Generate personalized recommendations

        Args:
            class_distribution: Distribution of classes
            class_percentage: Percentage of each class

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for each compensation pattern
        if class_percentage.get('scapular_winging', 0) > 10:
            recommendations.append({
                'issue': 'Scapular Winging',
                'severity': 'high' if class_percentage['scapular_winging'] > 30 else 'medium',
                'suggestion': 'Focus on scapular retraction exercises. Pull shoulder blades down and back during pushups.',
                'exercises': ['Wall slides', 'Scapular push-ups', 'Band pull-aparts']
            })

        if class_percentage.get('hip_sagging', 0) > 10:
            recommendations.append({
                'issue': 'Hip Sagging',
                'severity': 'high' if class_percentage['hip_sagging'] > 30 else 'medium',
                'suggestion': 'Strengthen core muscles. Maintain a plank position throughout the movement.',
                'exercises': ['Planks', 'Dead bugs', 'Hollow body holds']
            })

        if class_percentage.get('hip_piking', 0) > 10:
            recommendations.append({
                'issue': 'Hip Piking',
                'severity': 'high' if class_percentage['hip_piking'] > 30 else 'medium',
                'suggestion': 'Work on maintaining neutral spine alignment. Avoid raising hips too high.',
                'exercises': ['Plank variations', 'Bird dogs', 'Hip flexor stretches']
            })

        if class_percentage.get('elbow_flaring', 0) > 10:
            recommendations.append({
                'issue': 'Elbow Flaring',
                'severity': 'high' if class_percentage['elbow_flaring'] > 30 else 'medium',
                'suggestion': 'Keep elbows closer to body (< 45 degrees). This reduces shoulder strain.',
                'exercises': ['Narrow pushups', 'Tricep pushups', 'Close-grip bench press']
            })

        if class_percentage.get('trunk_instability', 0) > 10:
            recommendations.append({
                'issue': 'Trunk Instability',
                'severity': 'high' if class_percentage['trunk_instability'] > 30 else 'medium',
                'suggestion': 'Improve core stability and control. Maintain steady breathing.',
                'exercises': ['Anti-rotation exercises', 'Pallof press', 'Stability ball planks']
            })

        if class_percentage.get('correct_form', 0) > 80:
            recommendations.append({
                'issue': 'Excellent Form',
                'severity': 'positive',
                'suggestion': 'Great job! Consider progressing to more challenging variations.',
                'exercises': ['Decline pushups', 'Archer pushups', 'One-arm progressions']
            })

        return recommendations

    def _save_results(self, results: dict, video_path: str, participant_id: str):
        """
        Save analysis results

        Args:
            results: Analysis results
            video_path: Path to video
            participant_id: Participant ID
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        participant_id = participant_id or 'unknown'

        output_dir = os.path.join(config.RESULTS_DIR, 'analysis', participant_id)
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON report
        json_path = os.path.join(output_dir, f'analysis_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Analysis saved to: {json_path}")

        # Save CSV of frame data
        if 'detailed_frame_data' in results and len(results['detailed_frame_data']) > 0:
            df = pd.DataFrame(results['detailed_frame_data'])
            csv_path = os.path.join(output_dir, f'frame_data_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Frame data saved to: {csv_path}")

        # Generate summary report (text)
        self._save_text_report(results, output_dir, timestamp)

    def _save_text_report(self, results: dict, output_dir: str, timestamp: str):
        """
        Save human-readable text report

        Args:
            results: Analysis results
            output_dir: Output directory
            timestamp: Timestamp string
        """
        report_path = os.path.join(output_dir, f'report_{timestamp}.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXERCISE PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            f.write("METADATA\n")
            f.write("-" * 80 + "\n")
            for key, value in results['metadata'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # Performance Summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            for key, value in results['performance_summary'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # Class Distribution
            f.write("FORM DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            for class_name, percentage in results['class_distribution']['percentages'].items():
                count = results['class_distribution']['counts'][class_name]
                f.write(f"{class_name}: {count} frames ({percentage:.1f}%)\n")
            f.write("\n")

            # Temporal Analysis
            f.write("TEMPORAL ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for key, value in results['temporal_analysis'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            if len(results['recommendations']) == 0:
                f.write("No specific recommendations. Keep up the good work!\n")
            else:
                for i, rec in enumerate(results['recommendations'], 1):
                    f.write(f"\n{i}. {rec['issue']} (Severity: {rec['severity']})\n")
                    f.write(f"   {rec['suggestion']}\n")
                    f.write(f"   Recommended exercises: {', '.join(rec['exercises'])}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Text report saved to: {report_path}")

    def reset(self):
        """Reset analyzer state"""
        self.feature_buffer.clear()
        self.prediction_buffer.clear()
        self.feature_extractor.reset()


def batch_analyze(video_directory: str, model_path: str, output_dir: str = None):
    """
    Analyze multiple videos in batch

    Args:
        video_directory: Directory containing videos
        model_path: Path to trained model
        output_dir: Output directory for results
    """
    analyzer = PerformanceAnalyzer(model_path)

    video_files = [f for f in os.listdir(video_directory)
                  if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]

    print(f"Found {len(video_files)} videos to analyze")

    all_results = []

    for video_file in video_files:
        video_path = os.path.join(video_directory, video_file)
        participant_id = os.path.splitext(video_file)[0]

        print(f"\n{'='*80}")
        print(f"Analyzing: {video_file}")
        print(f"{'='*80}")

        try:
            results = analyzer.analyze_video(video_path, participant_id, save_results=True)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {video_file}: {e}")
            continue

        analyzer.reset()

    # Generate summary across all videos
    if output_dir and len(all_results) > 0:
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBatch summary saved to: {summary_path}")


def main():
    """
    Main analysis script
    """
    import argparse

    parser = argparse.ArgumentParser(description='Analyze exercise performance')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (for single analysis)')
    parser.add_argument('--batch', type=str, default=None,
                       help='Directory containing videos (for batch analysis)')
    parser.add_argument('--participant', type=str, default=None,
                       help='Participant ID')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')

    args = parser.parse_args()

    if args.batch:
        # Batch analysis
        batch_analyze(args.batch, args.model, args.output)
    elif args.video:
        # Single video analysis
        analyzer = PerformanceAnalyzer(args.model)
        results = analyzer.analyze_video(args.video, args.participant, save_results=True)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Form Quality Score: {results['performance_summary']['form_quality_score']:.1f}%")
        print(f"Total Reps: {results['performance_summary']['total_reps']}")
        print(f"Dominant Pattern: {results['performance_summary']['dominant_pattern']}")
    else:
        parser.error("Please specify either --video or --batch")


if __name__ == '__main__':
    main()
