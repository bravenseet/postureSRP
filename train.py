"""
Training Pipeline for Compensatory Movement Detection
Trains a Bidirectional LSTM model to classify movement patterns
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import config
from utils import set_seed, calculate_metrics, augment_sequence


class BiLSTMModel:
    """
    Bidirectional LSTM model for movement pattern classification
    """

    def __init__(self, sequence_length: int = None, feature_dim: int = None,
                 num_classes: int = None):
        """
        Initialize BiLSTM model

        Args:
            sequence_length: Length of input sequences
            feature_dim: Number of features per timestep
            num_classes: Number of output classes
        """
        self.sequence_length = sequence_length or config.SEQUENCE_LENGTH
        self.feature_dim = feature_dim or config.FEATURE_DIM
        self.num_classes = num_classes or config.NUM_CLASSES

        self.model = None
        self.history = None
        self.config = config.BILSTM_CONFIG

    def build_model(self) -> keras.Model:
        """
        Build the BiLSTM architecture

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim), name='input')

        # First Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.config['lstm_units_1'],
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(0.01)
            ),
            name='bilstm_1'
        )(inputs)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)

        # Second Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.config['lstm_units_2'],
                return_sequences=False,
                kernel_regularizer=keras.regularizers.l2(0.01)
            ),
            name='bilstm_2'
        )(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)

        # Dense layers
        x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.Dropout(self.config['dropout_rate'] / 2)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='BiLSTM_Posture')

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             class_weights: dict = None,
             use_augmentation: bool = True) -> keras.callbacks.History:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            class_weights: Dictionary of class weights for imbalanced data
            use_augmentation: Whether to use data augmentation

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        print(self.model.summary())

        # Data augmentation
        if use_augmentation:
            X_train_aug, y_train_aug = self._augment_data(X_train, y_train)
            print(f"Augmented training data: {X_train.shape} -> {X_train_aug.shape}")
            X_train = X_train_aug
            y_train = y_train_aug

        # Callbacks
        callbacks = self._create_callbacks()

        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return self.history

    def _augment_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Apply data augmentation

        Args:
            X: Input features
            y: Labels

        Returns:
            Augmented X and y
        """
        X_augmented = [X]
        y_augmented = [y]

        # Create augmented copies
        for i in range(len(X)):
            # Augment each sequence
            aug_sequence = augment_sequence(X[i], config.AUGMENTATION_CONFIG)
            X_augmented.append(aug_sequence[np.newaxis, :, :])
            y_augmented.append(y[i:i+1])

        X_aug = np.vstack(X_augmented)
        y_aug = np.concatenate(y_augmented)

        # Shuffle
        indices = np.random.permutation(len(X_aug))
        X_aug = X_aug[indices]
        y_aug = y_aug[indices]

        return X_aug, y_aug

    def _create_callbacks(self) -> list:
        """
        Create training callbacks

        Returns:
            List of Keras callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),

            # Model checkpoint
            ModelCheckpoint(
                filepath=os.path.join(config.MODEL_DIR, f'best_model_{timestamp}.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),

            # TensorBoard
            TensorBoard(
                log_dir=os.path.join(config.LOGS_DIR, f'tensorboard_{timestamp}'),
                histogram_freq=1
            )
        ]

        return callbacks

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test set

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, self.num_classes)

        # Add Keras evaluation metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)

        metrics['test_loss'] = test_loss
        metrics['keras_accuracy'] = test_acc
        metrics['keras_precision'] = test_precision
        metrics['keras_recall'] = test_recall

        return metrics

    def save_model(self, filepath: str = None):
        """
        Save the trained model

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(config.MODEL_DIR, f'bilstm_model_{timestamp}.keras')

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

        # Save training history
        if self.history is not None:
            history_path = filepath.replace('.keras', '_history.json')
            history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)

        return filepath

    def load_model(self, filepath: str):
        """
        Load a saved model

        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")

    def plot_training_history(self, save_path: str = None):
        """
        Plot training history

        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            raise ValueError("No training history available")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix

    Args:
        conf_matrix: Confusion matrix
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=list(config.MOVEMENT_CLASSES.values()),
        yticklabels=list(config.MOVEMENT_CLASSES.values())
    )

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """
    Main training pipeline
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train BiLSTM model for posture detection')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed data (.npz file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for model and results')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Fraction of training data to use for validation')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load data
    print(f"Loading data from: {args.data}")
    data = np.load(args.data)
    X = data['X']
    y = data['y']

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution:")
    for class_id in range(config.NUM_CLASSES):
        count = np.sum(y == class_id)
        print(f"  {config.MOVEMENT_CLASSES[class_id]}: {count} ({count/len(y)*100:.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Compute class weights for imbalanced data
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

    print(f"\nClass weights: {class_weights}")

    # Create and train model
    model = BiLSTMModel()
    model.build_model()

    # Train
    model.train(
        X_train, y_train,
        X_val, y_val,
        class_weights=class_weights,
        use_augmentation=not args.no_augmentation
    )

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = model.evaluate(X_test, y_test)

    print("\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")

    print("\nPer-class metrics:")
    for class_id, class_name in config.MOVEMENT_CLASSES.items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision_per_class'][class_id]:.4f}")
        print(f"    Recall: {metrics['recall_per_class'][class_id]:.4f}")
        print(f"    F1-Score: {metrics['f1_per_class'][class_id]:.4f}")

    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(config.RESULTS_DIR, f'training_{timestamp}')

    os.makedirs(args.output, exist_ok=True)

    # Save model
    model_path = os.path.join(args.output, 'final_model.keras')
    model.save_model(model_path)

    # Plot training history
    history_plot_path = os.path.join(args.output, 'training_history.png')
    model.plot_training_history(history_plot_path)

    # Plot confusion matrix
    conf_matrix_path = os.path.join(args.output, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], conf_matrix_path)

    # Save metrics
    metrics_path = os.path.join(args.output, 'metrics.json')
    metrics_to_save = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'test_loss': float(metrics['test_loss']),
        'per_class_metrics': {
            config.MOVEMENT_CLASSES[i]: {
                'precision': float(metrics['precision_per_class'][i]),
                'recall': float(metrics['recall_per_class'][i]),
                'f1': float(metrics['f1_per_class'][i])
            }
            for i in range(config.NUM_CLASSES)
        },
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
