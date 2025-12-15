"""
Demo Script for Compensatory Movement Detection System
Demonstrates the complete workflow from data collection to inference
"""
import os
import sys
import numpy as np
from datetime import datetime


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_step(step_num, text):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 80)


def check_requirements():
    """Check if all required packages are installed"""
    print_step(1, "Checking Requirements")

    required_packages = [
        'cv2',
        'mediapipe',
        'tensorflow',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]

    missing = []

    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mediapipe':
                import mediapipe
            elif package == 'tensorflow':
                import tensorflow
            elif package == 'numpy':
                import numpy
            elif package == 'pandas':
                import pandas
            elif package == 'sklearn':
                import sklearn
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'seaborn':
                import seaborn

            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed")
            missing.append(package)

    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False

    print("\n✓ All requirements satisfied!")
    return True


def check_directory_structure():
    """Check and create required directories"""
    print_step(2, "Checking Directory Structure")

    import config

    directories = [
        config.DATA_DIR,
        config.RAW_VIDEO_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODEL_DIR,
        config.RESULTS_DIR,
        config.LOGS_DIR
    ]

    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory} exists")
        else:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created {directory}")

    print("\n✓ Directory structure ready!")
    return True


def demo_feature_extraction():
    """Demonstrate feature extraction"""
    print_step(3, "Demo: Feature Extraction")

    print("Creating a sample pose landmark...")

    import mediapipe as mp
    from feature_extractor import FeatureExtractor

    # Create dummy landmarks (in practice, these come from MediaPipe)
    mp_pose = mp.solutions.pose

    # We'll create a simple standing pose for demonstration
    feature_extractor = FeatureExtractor()

    print(f"✓ Feature extractor initialized")
    print(f"✓ Expected feature dimension: {feature_extractor.feature_dim}")

    # Note: Real landmarks would come from processing a video frame
    print("\nTo extract features from a real video, use:")
    print("  python data_collector.py --visualize path/to/video.mp4")

    return True


def demo_data_structure():
    """Show expected data structure"""
    print_step(4, "Expected Data Structure")

    print("Organize your participant videos as follows:\n")
    print("data/raw_videos/")
    print("├── participant_01/")
    print("│   ├── correct_form/")
    print("│   │   ├── video_001.mp4")
    print("│   │   └── video_002.mp4")
    print("│   ├── scapular_winging/")
    print("│   │   └── video_001.mp4")
    print("│   ├── hip_sagging/")
    print("│   │   └── video_001.mp4")
    print("│   ├── hip_piking/")
    print("│   │   └── video_001.mp4")
    print("│   ├── elbow_flaring/")
    print("│   │   └── video_001.mp4")
    print("│   └── trunk_instability/")
    print("│       └── video_001.mp4")
    print("└── participant_02/")
    print("    └── ...")

    print("\nMovement classes:")
    import config
    for class_id, class_name in config.MOVEMENT_CLASSES.items():
        print(f"  {class_id}: {class_name}")

    return True


def create_sample_data():
    """Create sample synthetic data for testing"""
    print_step(5, "Creating Sample Data (for testing)")

    import config
    from utils import set_seed

    set_seed(42)

    # Create synthetic data
    num_samples = 100
    sequence_length = config.SEQUENCE_LENGTH
    feature_dim = config.FEATURE_DIM
    num_classes = config.NUM_CLASSES

    print(f"Generating {num_samples} synthetic sequences...")

    X_sample = np.random.randn(num_samples, sequence_length, feature_dim).astype(np.float32)
    y_sample = np.random.randint(0, num_classes, num_samples)

    # Save sample data
    sample_path = os.path.join(config.PROCESSED_DATA_DIR, 'sample_data.npz')
    np.savez_compressed(sample_path, X=X_sample, y=y_sample)

    print(f"✓ Sample data created: {sample_path}")
    print(f"  Shape: X={X_sample.shape}, y={y_sample.shape}")

    return sample_path


def demo_model_architecture():
    """Show model architecture"""
    print_step(6, "Model Architecture")

    from train import BiLSTMModel
    import config

    model = BiLSTMModel()
    model.build_model()

    print("BiLSTM Model Summary:\n")
    model.model.summary()

    print(f"\nModel Parameters:")
    print(f"  Sequence Length: {config.SEQUENCE_LENGTH}")
    print(f"  Feature Dimension: {config.FEATURE_DIM}")
    print(f"  Number of Classes: {config.NUM_CLASSES}")
    print(f"  LSTM Units (Layer 1): {config.BILSTM_CONFIG['lstm_units_1']}")
    print(f"  LSTM Units (Layer 2): {config.BILSTM_CONFIG['lstm_units_2']}")
    print(f"  Dropout Rate: {config.BILSTM_CONFIG['dropout_rate']}")

    return True


def print_usage_examples():
    """Print usage examples"""
    print_step(7, "Usage Examples")

    print("1. COLLECT DATA FROM VIDEOS:")
    print("   python data_collector.py --output data/processed/training_data.npz\n")

    print("2. VISUALIZE POSE DETECTION:")
    print("   python data_collector.py --visualize path/to/video.mp4\n")

    print("3. TRAIN MODEL:")
    print("   python train.py --data data/processed/training_data.npz --output results/training_001\n")

    print("4. TEST MODEL:")
    print("   python test.py --model results/training_001/final_model.keras \\")
    print("                  --data data/processed/training_data.npz\n")

    print("5. REAL-TIME INFERENCE (Webcam):")
    print("   python inference.py --model results/training_001/final_model.keras --source webcam\n")

    print("6. PROCESS VIDEO FILE:")
    print("   python inference.py --model results/training_001/final_model.keras \\")
    print("                       --source input.mp4 --output output.mp4\n")

    print("7. ANALYZE PERFORMANCE:")
    print("   python analyse.py --model results/training_001/final_model.keras \\")
    print("                     --video path/to/video.mp4 --participant P01\n")


def print_next_steps():
    """Print next steps"""
    print_header("Next Steps")

    print("1. READ DOCUMENTATION:")
    print("   - README.md: Project overview")
    print("   - SETUP_GUIDE.md: Detailed setup instructions")
    print("   - RESEARCH_NOTES.md: Research methodology\n")

    print("2. PREPARE DATA:")
    print("   - Record participant videos following the protocol")
    print("   - Organize videos in the expected structure")
    print("   - Aim for 10-15 participants, 10-15 reps each\n")

    print("3. COLLECT AND PROCESS DATA:")
    print("   - Process videos to extract features")
    print("   - Verify data quality and distribution\n")

    print("4. TRAIN MODEL:")
    print("   - Train BiLSTM model on collected data")
    print("   - Monitor training progress")
    print("   - Evaluate on test set\n")

    print("5. CONDUCT RESEARCH:")
    print("   - Compare ML vs rule-based detection")
    print("   - Analyze per-class performance")
    print("   - Compile results for report\n")

    print("6. DOCUMENTATION:")
    print("   - Document findings")
    print("   - Create visualizations")
    print("   - Prepare presentation/poster\n")


def main():
    """Main demo function"""
    print_header("Compensatory Movement Detection System - Demo")

    print("This demo will guide you through the system setup and usage.\n")

    # Check requirements
    if not check_requirements():
        print("\n⚠ Please install missing requirements first.")
        return

    # Check directory structure
    check_directory_structure()

    # Demo feature extraction
    demo_feature_extraction()

    # Show expected data structure
    demo_data_structure()

    # Create sample data
    sample_data_path = create_sample_data()

    # Show model architecture
    demo_model_architecture()

    # Print usage examples
    print_usage_examples()

    # Print next steps
    print_next_steps()

    print_header("Demo Complete!")

    print("The system is ready to use!")
    print("\nQuick test commands:")
    print(f"  1. Train on sample data: python train.py --data {sample_data_path}")
    print(f"  2. Test on sample data: python test.py --model [model_path] --data {sample_data_path}")
    print("\nFor full functionality, collect real participant videos and follow the setup guide.")
    print("\nGood luck with your research!\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
