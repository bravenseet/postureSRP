# Detailed Setup Guide

This guide provides step-by-step instructions for setting up and running the Compensatory Movement Detection System.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Data Collection](#data-collection)
4. [Training the Model](#training-the-model)
5. [Testing and Evaluation](#testing-and-evaluation)
6. [Running Inference](#running-inference)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware

- **CPU**: Multi-core processor (Intel i5 or equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)
- **Storage**: 10GB free space
- **Webcam**: For real-time inference (1080p recommended)

### Software

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8, 3.9, or 3.10 (3.11+ may have compatibility issues)
- **Camera**: Built-in or USB webcam for real-time testing

## Installation

### Step 1: Install Python

If you don't have Python installed:

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer and check "Add Python to PATH"
3. Verify: `python --version`

**macOS:**
```bash
brew install python@3.10
```

**Linux (Ubuntu):**
```bash
sudo apt update
sudo apt install python3.10 python3-pip python3-venv
```

### Step 2: Create Virtual Environment

```bash
# Navigate to project directory
cd postureV1

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# For GPU support (optional, NVIDIA GPU required)
pip install tensorflow-gpu==2.15.0
```

### Step 4: Verify Installation

```bash 
# Test imports
python -c "import cv2, mediapipe, tensorflow; print('All imports successful!')"
```

## Data Collection

### Video Recording Guidelines

#### Setup

1. **Camera Position**:
   - Place camera 6-8 feet from exercise area
   - Camera should be perpendicular to body (side view for pushups)
   - Ensure full body is visible in frame
   - Camera should be stationary (use tripod if possible)

2. **Lighting**:
   - Good, even lighting
   - Avoid backlighting (bright windows behind subject)
   - No harsh shadows

3. **Background**:
   - Plain, contrasting background
   - Avoid cluttered or busy backgrounds
   - Solid color wall works best

#### Recording Protocol

For each participant:

1. **Correct Form** (10-15 reps):
   - Demonstrate proper pushup technique
   - Body straight from head to heels
   - Elbows at 45 degrees
   - Full range of motion

2. **Scapular Winging** (10-15 reps):
   - Allow shoulder blades to protrude/wing out
   - Common in those with weak serratus anterior

3. **Hip Sagging** (10-15 reps):
   - Allow hips to drop toward ground
   - Indicates weak core

4. **Hip Piking** (10-15 reps):
   - Raise hips too high
   - Creates inverted V shape

5. **Elbow Flaring** (10-15 reps):
   - Elbows pointing out (>45 degrees from body)
   - Common compensation pattern

6. **Trunk Instability** (10-15 reps):
   - Allow torso to rotate or sway
   - Lack of core control

### Organizing Video Files

Create the following directory structure:

```
data/raw_videos/
├── participant_01/
│   ├── correct_form/
│   │   ├── video_001.mp4
│   │   └── video_002.mp4
│   ├── scapular_winging/
│   │   └── video_001.mp4
│   ├── hip_sagging/
│   │   └── video_001.mp4
│   ├── hip_piking/
│   │   └── video_001.mp4
│   ├── elbow_flaring/
│   │   └── video_001.mp4
│   └── trunk_instability/
│       └── video_001.mp4
├── participant_02/
│   └── [same structure]
└── ...
```

**Naming Conventions**:
- Participants: `participant_01`, `participant_02`, etc.
- Videos: `video_001.mp4`, `video_002.mp4`, etc.
- Supported formats: `.mp4`, `.avi`, `.mov`

### Processing Videos

#### Visualize Pose Detection (Optional)

Before processing all videos, test pose detection on a sample:

```bash
python data_collector.py --visualize data/raw_videos/participant_01/correct_form/video_001.mp4
```

This will create a video showing the detected pose landmarks.

#### Process All Videos

```bash
python data_collector.py --video_dir data/raw_videos --output data/processed/training_data.npz
```

**Expected Output**:
- Feature sequences saved to `training_data.npz`
- Metadata saved to `training_data_metadata.json`
- Console output showing processing progress and statistics

**Processing Time**: ~1-2 minutes per minute of video (depending on hardware)

## Training the Model

### Basic Training

```bash
python train.py \
    --data data/processed/training_data.npz \
    --output results/training_run_001
```

### Training Parameters

Customize training with these options:

```bash
python train.py \
    --data data/processed/training_data.npz \
    --output results/my_training \
    --test_size 0.2 \              # 20% for testing
    --val_size 0.15 \              # 15% for validation
    --no_augmentation \            # Disable data augmentation
    --seed 42                      # Random seed for reproducibility
```

```
python train.py --data data/processed/training_data.npz --output results/my_training --test_size 0.2 --val_size 0.15 --no_augmentation --seed 42                      
```

### Monitoring Training

Training progress is displayed in the console and saved to TensorBoard logs:

```bash
# View training progress in TensorBoard
tensorboard --logdir logs/
```

Open browser to `http://localhost:6006`

### Training Output

After training completes, you'll find:

```
results/training_run_001/
├── final_model.keras              # Trained model
├── final_model_history.json       # Training history
├── best_model_[timestamp].keras   # Best model during training
├── training_history.png           # Training curves
├── confusion_matrix.png           # Confusion matrix
└── metrics.json                   # Evaluation metrics
```

### Expected Training Time

- **CPU only**: 1-2 hours (for ~1000 sequences, 100 epochs)
- **GPU (NVIDIA)**: 10-20 minutes

### Interpreting Results

**Good Signs**:
- Validation accuracy > 85%
- Training and validation curves converge
- No overfitting (train/val gap < 5%)
- Balanced per-class F1 scores

**Warning Signs**:
- Large gap between train/val accuracy (overfitting)
- Validation loss increasing (early stopping should trigger)
- Very low accuracy on specific classes (data imbalance)

## Testing and Evaluation

### Run Comprehensive Tests

```bash
python test.py \
    --model results/training_run_001/final_model.keras \
    --data data/processed/training_data.npz \
    --output results/test_results_001
```

### Test Output

```
results/test_results_001/
├── test_results.json              # Detailed metrics
├── test_report.txt                # Human-readable report
├── accuracy_comparison.png        # ML vs Rule-based comparison
├── per_class_f1_comparison.png   # Per-class performance
└── confusion_matrices.png         # Side-by-side confusion matrices
```

### Understanding Test Results

The test report includes:

1. **Overall Accuracy**: ML vs rule-based comparison
2. **Statistical Significance**: McNemar's test, Cohen's h
3. **Per-Class Metrics**: Precision, Recall, F1 for each class
4. **Confusion Matrices**: Visual representation of predictions

**Target Metrics**:
- Overall Accuracy: >90%
- Per-class F1: >0.85
- Improvement over rule-based: >15%

## Running Inference

### Real-time Webcam Inference

```bash
# Default webcam
python inference.py --model results/training_run_001/final_model.keras --source webcam

# Specific camera (if multiple cameras)
python inference.py --model results/training_run_001/final_model.keras --source 0
```

**Controls**:
- Press `q` to quit
- Press `r` to reset rep counter

### Process Video File

```bash
python inference.py \
    --model results/training_run_001/final_model.keras \
    --source path/to/input/video.mp4 \
    --output path/to/output/video.mp4
```

### Analyze Without Feedback

For research comparison (participants perform without seeing AI feedback):

```bash
# Single video
python analyse.py \
    --model results/training_run_001/final_model.keras \
    --video path/to/video.mp4 \
    --participant P01

# Batch analysis
python analyse.py \
    --model results/training_run_001/final_model.keras \
    --batch data/analysis_videos/ \
    --output results/analysis_001
```

## Advanced Configuration

### Adjusting Model Parameters

Edit `config.py` to customize:

```python
# Model architecture
BILSTM_CONFIG = {
    'lstm_units_1': 128,      # First LSTM layer units
    'lstm_units_2': 64,       # Second LSTM layer units
    'dropout_rate': 0.3,      # Dropout rate
    'learning_rate': 0.001,   # Learning rate
    'batch_size': 32,         # Batch size
    'epochs': 100,            # Max epochs
}

# Sequence length (frames)
SEQUENCE_LENGTH = 30

# MediaPipe settings
MEDIAPIPE_MODEL_COMPLEXITY = 2  # 0=fastest, 2=most accurate
```

### Optimizing for Your Hardware

**CPU-only (slower but works everywhere)**:
```python
# In config.py, reduce complexity
MEDIAPIPE_MODEL_COMPLEXITY = 1
BILSTM_CONFIG['batch_size'] = 16
```

**GPU (faster)**:
```python
# Install GPU version
pip install tensorflow-gpu==2.15.0

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Troubleshooting

### Common Issues

#### 1. "No pose detected" during inference

**Solutions**:
- Ensure good lighting
- Move farther from camera (full body visible)
- Adjust `MEDIAPIPE_MIN_DETECTION_CONFIDENCE` in `config.py` (lower value)
- Check camera permissions

#### 2. Low model accuracy

**Solutions**:
- Collect more data (aim for 100+ samples per class)
- Balance dataset (equal samples per class)
- Increase `SEQUENCE_LENGTH` for more temporal context
- Try different augmentation settings
- Check video quality and pose detection

#### 3. Slow inference

**Solutions**:
- Reduce `MEDIAPIPE_MODEL_COMPLEXITY` to 1 or 0
- Skip frames: adjust `skip_frames` in `config.py`
- Use GPU acceleration
- Reduce video resolution

#### 4. Import/Installation errors

```bash
# Clear cache and reinstall
pip cache purge
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

#### 5. Out of memory during training

**Solutions**:
- Reduce `batch_size` in `config.py`
- Use smaller `SEQUENCE_LENGTH`
- Process videos in smaller batches
- Close other applications

### Getting Help

If you encounter issues:

1. Check console output for error messages
2. Review `logs/` directory for detailed logs
3. Verify your data structure matches expected format
4. Ensure all dependencies are correctly installed
5. Check Python version compatibility

## Next Steps

After setup:

1. **Collect initial data**: Start with 2-3 participants
2. **Test pipeline**: Process data and train a preliminary model
3. **Evaluate results**: Check if detection works reasonably
4. **Scale up**: Collect data from all 10-15 participants
5. **Final training**: Train production model on full dataset
6. **Research analysis**: Run comprehensive tests and comparisons
7. **Document findings**: Compile results for your research report

## Validation Checklist

Before proceeding with research:

- [ ] All dependencies installed
- [ ] Pose detection works on sample video
- [ ] Directory structure created
- [ ] At least 2 participant videos collected
- [ ] Data processing successful
- [ ] Model trains without errors
- [ ] Inference works on webcam
- [ ] Test comparison generates results

Good luck with your research project!
