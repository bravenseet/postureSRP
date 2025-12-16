# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-based compensatory movement detection system for physiotherapy exercises (specifically pushups), developed as a SSEF science research project. The system uses MediaPipe for pose estimation and BiLSTM neural networks to classify movements into 6 categories: correct form and 5 compensatory patterns (scapular winging, hip sagging, hip piking, elbow flaring, trunk instability). Target accuracy: 90%+.

## Common Commands

### Data Collection & Processing
```bash
# Process all participant videos
python data_collector.py --output data/processed/training_data.npz

# Visualize pose detection on sample video
python data_collector.py --visualize data/raw_videos/participant_01/correct_form/video_001.mp4
```

### Training
```bash
# Train BiLSTM model with default parameters
python train.py --data data/processed/training_data.npz --output results/training_001

# Train with custom parameters
python train.py --data data/processed/training_data.npz \
                --output results/training_001 \
                --test_size 0.2 --val_size 0.15 --seed 42

# Monitor training with TensorBoard
tensorboard --logdir logs/
```

### Testing & Evaluation
```bash
# Comprehensive test (ML vs rule-based comparison)
python test.py --model results/training_001/final_model.keras \
               --data data/processed/training_data.npz \
               --output results/test_001
```

### Real-time Inference
```bash
# Run on webcam with live feedback
python inference.py --model results/training_001/final_model.keras --source webcam

# Process video file
python inference.py --model results/training_001/final_model.keras \
                    --source input.mp4 --output output.mp4
```

### Analysis (without feedback for research)
```bash
# Single video analysis
python analyse.py --model results/training_001/final_model.keras \
                  --video path/to/video.mp4 --participant P01

# Batch analysis
python analyse.py --model results/training_001/final_model.keras \
                  --batch data/raw_videos/participant_01/
```

### Demo & Validation
```bash
# Run demo to verify setup
python demo.py
```

## Architecture

### Core Pipeline Flow
```
Video/Webcam Input
  ↓
MediaPipe Pose (33 landmarks)
  ↓
Feature Extractor (78 biomechanical features)
  ↓
Sequence Windows (30 frames)
  ↓
BiLSTM Model (128→64 units)
  ↓
6-class Classification
  ↓
Visual Feedback / Analysis Report
```

### Feature Extraction System (78 features)

The `FeatureExtractor` class extracts 6 categories of features per frame:
1. **Joint Angles (24)**: Elbow, shoulder, hip, knee, ankle, wrist angles (bilateral)
2. **Alignment (12)**: Shoulder/hip alignment, lateral deviation, symmetry
3. **Velocities (15)**: Joint velocities, center of mass velocity, angular velocities
4. **Distances (12)**: Inter-joint distances, body proportions
5. **Temporal (9)**: Rate of change, acceleration, movement smoothness
6. **Stability (6)**: Bilateral symmetry, postural sway, body stability

**Important**: As of recent updates, the system handles **variable feature dimensions** gracefully:
- `ALLOW_VARIABLE_FEATURES = True` in config.py
- Minimum 30 features required (`MIN_FEATURE_DIM`)
- Automatic padding/truncation to 78 features
- NaN/Inf values cleaned automatically
- Each feature extraction category has try-except protection

### Model Architecture

**BiLSTM Structure**:
```
Input: (batch, 30 frames, 78 features)
  ↓
BiLSTM Layer 1: 128 units (bidirectional)
  ↓ Batch Normalization + Dropout (0.3)
BiLSTM Layer 2: 64 units (bidirectional)
  ↓ Batch Normalization + Dropout (0.3)
Dense Layer: 64 units (ReLU)
  ↓ Dropout (0.15)
Output: 6 classes (Softmax)
```

### File Roles

**Core Modules**:
- `config.py`: All configuration parameters (model, MediaPipe, thresholds, paths)
- `utils.py`: Shared functions (angle/distance/velocity calculation, metrics, normalization)
- `feature_extractor.py`: 78-feature biomechanical extraction with error handling
- `data_collector.py`: Batch video processing, feature extraction, sequence generation
- `train.py`: BiLSTM training with augmentation, class balancing, callbacks
- `inference.py`: Real-time detection with visual feedback and rep counting
- `analyse.py`: Performance analysis WITHOUT feedback (for research comparison)
- `test.py`: Comprehensive evaluation, ML vs rule-based comparison, statistical tests

**Supporting**:
- `demo.py`: System verification and interactive demo
- `requirements.txt`: Python dependencies

## Key Design Patterns

### 1. Feature Robustness Pattern
All feature extraction uses safe calculation wrappers:
```python
def safe_angle(*args):
    try:
        angle = calculate_angle(*args)
        if np.isnan(angle) or np.isinf(angle):
            return 0.0
        return float(angle)
    except:
        return 0.0
```

Each feature category extraction is wrapped in try-except blocks, falling back to zero-padded features if extraction fails. This ensures the system continues running even with partial pose detection.

### 2. Temporal Sequence Pattern
- Videos → Frame-by-frame features → Sliding windows (30 frames, 50% overlap)
- Maintains temporal context for BiLSTM
- Previous landmarks/features stored in `FeatureExtractor` for velocity calculations

### 3. Dual Mode Pattern
- **inference.py**: Real-time feedback (for clinical/training use)
- **analyse.py**: No feedback (for research validation - tests detection accuracy without influencing participant behavior)

### 4. State Management
Both `RealtimeInference` and `PerformanceAnalyzer` classes maintain:
- Feature buffer (deque, length=SEQUENCE_LENGTH)
- Prediction buffer (for temporal smoothing)
- Rep counter state (up/down transitions)
- Previous frame data (for velocity calculations)

Remember to call `.reset()` between videos to clear state.

## Critical Implementation Details

### Data Directory Structure
The system expects this exact structure:
```
data/raw_videos/
├── participant_01/
│   ├── correct_form/
│   ├── scapular_winging/
│   ├── hip_sagging/
│   ├── hip_piking/
│   ├── elbow_flaring/
│   └── trunk_instability/
└── participant_02/
    └── ...
```

Class labels are automatically mapped from directory names via `config.MOVEMENT_CLASSES`.

### Feature Dimension Handling
**Recent Update**: The system now gracefully handles incomplete feature extraction:
- Check `config.ALLOW_VARIABLE_FEATURES`, `MIN_FEATURE_DIM`, `FEATURE_PADDING_VALUE`
- All feature extraction methods have error handling
- `_clean_features()` method removes NaN/Inf values
- Padding/truncation happens automatically in feature_extractor, train.py, inference.py, and analyse.py
- Data validation occurs before adding to buffers

### Sequence Window Generation
In `data_collector.py`, use `utils.create_sequence_windows()` with:
- Window size: `config.SEQUENCE_LENGTH` (30 frames)
- Overlap: 0.5 (50% overlap between sequences)
- This generates training samples from continuous video

### Rep Counting Logic
Based on elbow angle thresholds:
- Transition to "down": angle < `REP_COUNTER['elbow_angle_threshold_down']` (100°)
- Complete rep: angle > `REP_COUNTER['elbow_angle_threshold_up']` (160°)
- Minimum frames between reps to prevent false counts

### Class Weights
Training uses `compute_class_weight('balanced')` to handle class imbalance. This is critical because participants may have varying numbers of samples per compensation pattern.

## Configuration Parameters

### Most Commonly Adjusted
In `config.py`:

```python
# Feature extraction
SEQUENCE_LENGTH = 30  # Frames per sequence (1 second at 30fps)
FEATURE_DIM = 78  # Feature vector size
MIN_FEATURE_DIM = 30  # Minimum acceptable features
ALLOW_VARIABLE_FEATURES = True  # Enable flexible feature handling

# Model architecture
BILSTM_CONFIG = {
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
}

# MediaPipe settings
MEDIAPIPE_MODEL_COMPLEXITY = 2  # 0=fast, 2=accurate
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.7

# Inference
INFERENCE_CONFIG = {
    'confidence_threshold': 0.7,
    'smoothing_window': 5,  # Frames to smooth predictions
}
```

### Performance Tuning
- **Slow inference**: Reduce `MEDIAPIPE_MODEL_COMPLEXITY` to 1 or 0
- **Low accuracy**: Increase `SEQUENCE_LENGTH`, collect more data, adjust augmentation
- **Out of memory**: Reduce `batch_size`
- **Unstable predictions**: Increase `smoothing_window`

## Testing Strategy

### Statistical Comparison Framework
`test.py` implements comprehensive evaluation:
1. **ML Model Evaluation**: Standard metrics (accuracy, precision, recall, F1)
2. **Rule-Based Baseline**: Fixed-threshold detection for comparison
3. **Statistical Tests**:
   - McNemar's test (paired predictions, p-value)
   - Cohen's h (effect size)
   - Per-class performance comparison

This validates the hypothesis that ML outperforms rule-based detection by >15%.

### Expected Metrics
- Overall Accuracy: Target ≥90%
- Per-class F1: Target ≥0.85
- ML vs Rule-based improvement: Target >15%

## Research Context

This is a SSEF science research project testing the hypothesis: "Can an AI system detect compensatory movement patterns during physiotherapy exercises more accurately than self-observation?"

**Novel Contribution**: Automatic detection and classification of 5 compensatory patterns using temporal biomechanical features and BiLSTM.

**Data Collection Protocol**:
- 10-15 participants
- 10-15 reps per form per participant
- Side-view video recording
- Total: ~900-1350 labeled repetitions

**Use analyse.py (not inference.py)** for research validation to avoid feedback bias.

## Important Constraints

1. **Single Exercise**: Only pushups analyzed (extensible to other exercises)
2. **2D Analysis**: Side-view only (no depth information)
3. **Controlled Environment**: Laboratory conditions with good lighting
4. **Sequential Processing**: Videos processed frame-by-frame (not batch)
5. **Temporal Dependencies**: Feature extractor maintains state (velocity calculations)

## Common Pitfalls to Avoid

1. **Don't forget to reset state**: Call `.reset()` on `FeatureExtractor`, `RealtimeInference`, or `PerformanceAnalyzer` between videos
2. **Don't skip data validation**: Loaded data may have different feature dimensions - always validate and pad/truncate
3. **Don't ignore class imbalance**: Always use class weights during training
4. **Don't mix inference modes**: Use `inference.py` for real-time feedback, `analyse.py` for research (no feedback)
5. **Don't assume 78 features**: With `ALLOW_VARIABLE_FEATURES=True`, accept ≥30 features and handle gracefully
6. **Don't train without augmentation**: Unless specifically testing, use augmentation to improve robustness
7. **Don't compare models without statistical tests**: Use McNemar's test for paired predictions

## Extension Points

To add new exercises or compensatory patterns:
1. Update `MOVEMENT_CLASSES` in config.py
2. Add corresponding directories in data structure
3. Adjust `RULE_BASED_THRESHOLDS` if using rule-based comparison
4. Consider if feature extraction needs exercise-specific modifications
5. Update feedback messages in `inference.py`

## Debugging Tips

- **Check feature dimensions**: Print `features.shape` after extraction
- **Visualize pose landmarks**: Use `data_collector.py --visualize`
- **Monitor training**: Use TensorBoard to track metrics
- **Validate data**: Check `metadata.json` for statistics
- **Test incrementally**: Start with 2-3 participants before full dataset
- **Check NaN/Inf**: The system handles these automatically now, but verify in logs
