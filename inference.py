"""
Real-time Inference for Compensatory Movement Detection
Provides live visual feedback during exercise performance
"""
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from collections import deque
import time
import config
from feature_extractor import FeatureExtractor
from utils import smooth_predictions, put_text_with_background


class RealtimeInference:
    """
    Real-time movement pattern detection with visual feedback
    """

    def __init__(self, model_path: str):
        """
        Initialize real-time inference

        Args:
            model_path: Path to trained model
        """
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            model_complexity=config.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Buffers for sequence collection
        self.feature_buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=config.INFERENCE_CONFIG['smoothing_window'])

        # Rep counter
        self.rep_count = 0
        self.rep_state = 'up'  # 'up' or 'down'
        self.last_rep_time = 0

        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)

        # Current prediction
        self.current_prediction = None
        self.current_confidence = 0.0

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame

        Args:
            frame: Input frame (BGR)

        Returns:
            annotated_frame: Frame with annotations
            prediction: Current prediction
            confidence: Prediction confidence
        """
        start_time = time.time()

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(frame_rgb)

        # Draw pose landmarks
        annotated_frame = frame.copy()

        if results.pose_landmarks:
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Extract features
            features = self.feature_extractor.extract_features(results.pose_landmarks)

            # Validate feature dimensions
            if len(features) == config.FEATURE_DIM:
                self.feature_buffer.append(features)
            else:
                # Still append but log warning
                if len(features) < config.MIN_FEATURE_DIM:
                    # Features too small, skip this frame
                    pass
                else:
                    # Features acceptable, append
                    self.feature_buffer.append(features)

            # Make prediction if buffer is full
            if len(self.feature_buffer) == config.SEQUENCE_LENGTH:
                prediction, confidence = self._predict()
                self.current_prediction = prediction
                self.current_confidence = confidence

                # Update rep counter
                self._update_rep_counter(features)

                # Draw feedback
                annotated_frame = self._draw_feedback(annotated_frame, prediction, confidence)
            else:
                # Buffer not full yet
                annotated_frame = self._draw_loading(annotated_frame)
        else:
            # No pose detected
            annotated_frame = self._draw_no_pose(annotated_frame)

        # Calculate FPS
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        self.fps_buffer.append(fps)

        # Draw FPS if enabled
        if config.INFERENCE_CONFIG['display_fps']:
            annotated_frame = self._draw_fps(annotated_frame, fps)

        return annotated_frame, self.current_prediction, self.current_confidence

    def _predict(self) -> tuple:
        """
        Make prediction from feature buffer

        Returns:
            prediction: Predicted class
            confidence: Prediction confidence
        """
        try:
            # Convert buffer to numpy array
            sequence = np.array(list(self.feature_buffer))

            # Validate sequence shape
            if sequence.shape[1] != config.FEATURE_DIM:
                # Pad or truncate features if needed
                if sequence.shape[1] < config.FEATURE_DIM:
                    padding = np.full((sequence.shape[0], config.FEATURE_DIM - sequence.shape[1]),
                                     config.FEATURE_PADDING_VALUE)
                    sequence = np.concatenate([sequence, padding], axis=1)
                else:
                    sequence = sequence[:, :config.FEATURE_DIM]

            # Clean NaN/Inf values
            sequence = np.nan_to_num(sequence, nan=config.FEATURE_PADDING_VALUE,
                                    posinf=1e6, neginf=-1e6)
            sequence = sequence[np.newaxis, :, :]  # Add batch dimension

            # Predict
            predictions = self.model.predict(sequence, verbose=0)
            confidence = np.max(predictions[0])
            prediction_class = np.argmax(predictions[0])
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0.0

        # Apply smoothing
        self.prediction_buffer.append(prediction_class)
        smoothed_prediction = smooth_predictions(
            list(self.prediction_buffer),
            config.INFERENCE_CONFIG['smoothing_window']
        )

        # Only return prediction if confidence is high enough
        if confidence < config.INFERENCE_CONFIG['confidence_threshold']:
            return None, confidence

        return smoothed_prediction, confidence

    def _update_rep_counter(self, features: np.ndarray):
        """
        Update repetition counter based on elbow angles

        Args:
            features: Current feature vector
        """
        # Elbow angles are the first two features
        avg_elbow_angle = (features[0] + features[1]) / 2

        current_time = time.time()

        # Prevent counting too quickly
        if current_time - self.last_rep_time < config.REP_COUNTER['min_frames_between_reps'] / 30.0:
            return

        threshold_down = config.REP_COUNTER['elbow_angle_threshold_down']
        threshold_up = config.REP_COUNTER['elbow_angle_threshold_up']

        if self.rep_state == 'up' and avg_elbow_angle < threshold_down:
            # Moving to down position
            self.rep_state = 'down'

        elif self.rep_state == 'down' and avg_elbow_angle > threshold_up:
            # Moving to up position - rep complete
            self.rep_count += 1
            self.rep_state = 'up'
            self.last_rep_time = current_time

    def _draw_feedback(self, frame: np.ndarray, prediction: int,
                       confidence: float) -> np.ndarray:
        """
        Draw visual feedback on frame

        Args:
            frame: Input frame
            prediction: Predicted class
            confidence: Prediction confidence

        Returns:
            Annotated frame
        """
        h, w, _ = frame.shape

        if prediction is None:
            # Low confidence
            text = "Uncertain"
            color = (128, 128, 128)  # Gray
        else:
            # Get class name and color
            class_name = config.MOVEMENT_CLASSES[prediction]
            color = config.FEEDBACK_COLORS[class_name]
            text = f"{class_name.replace('_', ' ').title()}"

        # Draw main feedback box
        box_height = 80
        cv2.rectangle(frame, (0, 0), (w, box_height), color, -1)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        text_x = (w - text_width) // 2
        text_y = (box_height + text_height) // 2

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Draw confidence
        conf_text = f"Confidence: {confidence:.1%}"
        conf_y = box_height + 30
        put_text_with_background(frame, conf_text, (10, conf_y), 0.6, (255, 255, 255), 2)

        # Draw rep counter
        if config.INFERENCE_CONFIG['display_rep_count']:
            rep_text = f"Reps: {self.rep_count}"
            rep_x = w - 150
            rep_y = box_height + 30
            put_text_with_background(frame, rep_text, (rep_x, rep_y), 0.8, (255, 255, 255), 2)

        # Draw corrective feedback
        if config.INFERENCE_CONFIG['display_feedback'] and prediction is not None:
            if prediction != 0:  # Not correct form
                feedback_text = self._get_feedback_message(prediction)
                feedback_y = h - 60

                # Draw semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                put_text_with_background(frame, feedback_text, (10, feedback_y), 0.7, (255, 255, 255), 2)
            else:
                # Good form!
                good_text = "Excellent Form! Keep it up!"
                good_y = h - 60
                put_text_with_background(frame, good_text, (10, good_y), 0.7, (0, 255, 0), 2)

        return frame

    def _get_feedback_message(self, prediction: int) -> str:
        """
        Get corrective feedback message for prediction

        Args:
            prediction: Predicted class

        Returns:
            Feedback message
        """
        feedback_messages = {
            1: "Scapular Winging: Engage shoulder blades, pull them down and back",
            2: "Hip Sagging: Engage core, keep hips in line with shoulders",
            3: "Hip Piking: Lower hips, maintain straight body line",
            4: "Elbow Flaring: Keep elbows closer to body, <45 degrees",
            5: "Trunk Instability: Engage core, maintain stable plank position"
        }

        return feedback_messages.get(prediction, "Form needs adjustment")

    def _draw_loading(self, frame: np.ndarray) -> np.ndarray:
        """Draw loading message"""
        h, w, _ = frame.shape

        buffer_progress = len(self.feature_buffer) / config.SEQUENCE_LENGTH
        text = f"Initializing... {buffer_progress:.0%}"

        put_text_with_background(frame, text, (10, 30), 0.7, (255, 255, 0), 2)

        return frame

    def _draw_no_pose(self, frame: np.ndarray) -> np.ndarray:
        """Draw no pose detected message"""
        text = "No pose detected - please position yourself in frame"
        put_text_with_background(frame, text, (10, 30), 0.7, (0, 0, 255), 2)

        return frame

    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter"""
        h, w, _ = frame.shape
        fps_text = f"FPS: {fps:.1f}"
        put_text_with_background(frame, fps_text, (w - 120, h - 20), 0.6, (255, 255, 255), 2)

        return frame

    def run_webcam(self, camera_id: int = 0):
        """
        Run inference on webcam feed

        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        print("Starting real-time inference...")
        print("Press 'q' to quit, 'r' to reset rep counter")

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # Process frame
                annotated_frame, prediction, confidence = self.process_frame(frame)

                # Display
                cv2.imshow('Posture Detection', annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.rep_count = 0
                    print("Rep counter reset")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nSession complete. Total reps: {self.rep_count}")

    def run_video(self, video_path: str, output_path: str = None):
        """
        Run inference on video file

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")

        # Reset state
        self.reset()

        frame_count = 0
        predictions_list = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # Process frame
                annotated_frame, prediction, confidence = self.process_frame(frame)

                # Store prediction
                predictions_list.append({
                    'frame': frame_count,
                    'prediction': int(prediction) if prediction is not None else None,
                    'confidence': float(confidence),
                    'class_name': config.MOVEMENT_CLASSES[prediction] if prediction is not None else None
                })

                # Write or display
                if writer:
                    writer.write(annotated_frame)
                else:
                    cv2.imshow('Posture Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")

        finally:
            cap.release()
            if writer:
                writer.release()
                print(f"Output saved to: {output_path}")
            else:
                cv2.destroyAllWindows()

            print(f"\nVideo processing complete. Total reps: {self.rep_count}")

        return predictions_list

    def reset(self):
        """Reset inference state"""
        self.feature_buffer.clear()
        self.prediction_buffer.clear()
        self.rep_count = 0
        self.rep_state = 'up'
        self.last_rep_time = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        self.feature_extractor.reset()


def main():
    """
    Main inference script
    """
    import argparse

    parser = argparse.ArgumentParser(description='Real-time posture detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--source', type=str, default='webcam',
                       help='Input source: "webcam", camera ID (e.g., "0"), or video path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (only for video files)')

    args = parser.parse_args()

    # Initialize inference
    inference = RealtimeInference(args.model)

    # Run based on source
    if args.source.lower() == 'webcam' or args.source.isdigit():
        camera_id = 0 if args.source.lower() == 'webcam' else int(args.source)
        inference.run_webcam(camera_id)
    else:
        # Video file
        inference.run_video(args.source, args.output)


if __name__ == '__main__':
    main()