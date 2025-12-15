"""
Testing and Evaluation Module
Comprehensive testing with quantitative benchmarks
Compares ML-based detection with rule-based detection
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            confusion_matrix, classification_report, roc_auc_score,
                            roc_curve, auc)
from scipy.stats import ttest_ind
import json
import os
from datetime import datetime
import config
from utils import calculate_angle, calculate_metrics


class RuleBasedDetector:
    """
    Rule-based compensatory pattern detector for comparison
    """

    def __init__(self):
        self.thresholds = config.RULE_BASED_THRESHOLDS

    def detect(self, features: np.ndarray) -> int:
        """
        Detect compensatory pattern using rule-based logic

        Args:
            features: Feature vector (78 features)

        Returns:
            Predicted class (0-5)
        """
        # Extract relevant features for rule-based detection
        # Features are ordered as defined in feature_extractor.py

        # Elbow angles (features 0-1)
        left_elbow_angle = features[0]
        right_elbow_angle = features[1]
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

        # Hip angles (features 4-5)
        left_hip_angle = features[4]
        right_hip_angle = features[5]
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2

        # Trunk angles (features 8-9)
        left_trunk_angle = features[8]
        right_trunk_angle = features[9]

        # Elbow flare angles (features 10-11)
        left_elbow_flare = features[10]
        right_elbow_flare = features[11]
        avg_elbow_flare = (left_elbow_flare + right_elbow_flare) / 2

        # Body angles (features 18-19)
        left_body_angle = features[18]
        right_body_angle = features[19]
        avg_body_angle = (left_body_angle + right_body_angle) / 2

        # Alignment features (features 24-35)
        shoulder_alignment = features[24]
        hip_height_ratio = features[27]

        # Stability features (features 72-77)
        shoulder_symmetry = features[72]
        postural_sway = features[76]

        # Apply rules in order of priority
        # Check for elbow flaring
        if avg_elbow_flare > self.thresholds['elbow_flaring']['elbow_torso_angle']:
            return 4  # elbow_flaring

        # Check for hip sagging
        if avg_hip_angle < self.thresholds['hip_sagging']['hip_angle']:
            return 2  # hip_sagging

        # Check for hip piking
        if avg_hip_angle > 175 and avg_body_angle < self.thresholds['hip_piking']['hip_angle']:
            return 3  # hip_piking

        # Check for scapular winging (using shoulder symmetry as proxy)
        if shoulder_symmetry > self.thresholds['scapular_winging']['scapular_elevation_diff'] / 100:
            return 1  # scapular_winging

        # Check for trunk instability
        if postural_sway > self.thresholds['trunk_instability']['lateral_trunk_deviation']:
            return 5  # trunk_instability

        # Default to correct form
        return 0  # correct_form


class ModelTester:
    """
    Comprehensive model testing and evaluation
    """

    def __init__(self, model_path: str):
        """
        Initialize tester

        Args:
            model_path: Path to trained model
        """
        self.ml_model = keras.models.load_model(model_path)
        self.rule_based_model = RuleBasedDetector()
        self.results = {}

    def test_models(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Test both ML and rule-based models

        Args:
            X_test: Test features (sequences)
            y_test: Test labels

        Returns:
            Comprehensive test results
        """
        print("Testing models...")
        print(f"Test set size: {len(X_test)} sequences")

        # ML Model predictions
        print("\nEvaluating ML model...")
        y_pred_ml_probs = self.ml_model.predict(X_test, verbose=1)
        y_pred_ml = np.argmax(y_pred_ml_probs, axis=1)

        # Rule-based predictions
        print("\nEvaluating rule-based model...")
        y_pred_rule = []

        for sequence in X_test:
            # Use middle frame of sequence for rule-based detection
            middle_frame = sequence[len(sequence) // 2]
            prediction = self.rule_based_model.detect(middle_frame)
            y_pred_rule.append(prediction)

        y_pred_rule = np.array(y_pred_rule)

        # Calculate metrics for both models
        ml_metrics = self._calculate_comprehensive_metrics(y_test, y_pred_ml, y_pred_ml_probs, "ML Model")
        rule_metrics = self._calculate_comprehensive_metrics(y_test, y_pred_rule, None, "Rule-Based Model")

        # Statistical comparison
        statistical_comparison = self._statistical_comparison(y_test, y_pred_ml, y_pred_rule)

        # Per-class analysis
        per_class_comparison = self._per_class_comparison(y_test, y_pred_ml, y_pred_rule)

        # Compile results
        self.results = {
            'test_info': {
                'test_size': len(X_test),
                'num_classes': config.NUM_CLASSES,
                'class_distribution': {
                    config.MOVEMENT_CLASSES[i]: int(np.sum(y_test == i))
                    for i in range(config.NUM_CLASSES)
                }
            },
            'ml_model': ml_metrics,
            'rule_based_model': rule_metrics,
            'statistical_comparison': statistical_comparison,
            'per_class_comparison': per_class_comparison
        }

        return self.results

    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_pred_probs: np.ndarray, model_name: str) -> dict:
        """
        Calculate comprehensive metrics for a model

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_probs: Prediction probabilities (optional)
            model_name: Name of the model

        Returns:
            Metrics dictionary
        """
        print(f"\nCalculating metrics for {model_name}...")

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=range(config.NUM_CLASSES)
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))

        # Sensitivity and specificity per class
        sensitivity = []
        specificity = []

        for i in range(config.NUM_CLASSES):
            tp = conf_matrix[i, i]
            fn = np.sum(conf_matrix[i, :]) - tp
            fp = np.sum(conf_matrix[:, i]) - tp
            tn = np.sum(conf_matrix) - tp - fn - fp

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            sensitivity.append(sens)
            specificity.append(spec)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class_metrics': {
                config.MOVEMENT_CLASSES[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'sensitivity': float(sensitivity[i]),
                    'specificity': float(specificity[i]),
                    'support': int(support[i]) if isinstance(support, np.ndarray) else int(np.sum(y_true == i))
                }
                for i in range(config.NUM_CLASSES)
            },
            'confusion_matrix': conf_matrix.tolist()
        }

        # ROC-AUC if probabilities available
        if y_pred_probs is not None:
            try:
                # One-vs-rest ROC-AUC
                roc_auc_per_class = {}
                for i in range(config.NUM_CLASSES):
                    if np.sum(y_true == i) > 0:  # Only if class exists in test set
                        y_true_binary = (y_true == i).astype(int)
                        y_score = y_pred_probs[:, i]
                        roc_auc = roc_auc_score(y_true_binary, y_score)
                        roc_auc_per_class[config.MOVEMENT_CLASSES[i]] = float(roc_auc)

                metrics['roc_auc_per_class'] = roc_auc_per_class
                metrics['avg_roc_auc'] = float(np.mean(list(roc_auc_per_class.values())))
            except Exception as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")

        return metrics

    def _statistical_comparison(self, y_true: np.ndarray, y_pred_ml: np.ndarray,
                                y_pred_rule: np.ndarray) -> dict:
        """
        Perform statistical comparison between models

        Args:
            y_true: True labels
            y_pred_ml: ML model predictions
            y_pred_rule: Rule-based predictions

        Returns:
            Statistical comparison results
        """
        # Per-sample correctness
        ml_correct = (y_pred_ml == y_true).astype(int)
        rule_correct = (y_pred_rule == y_true).astype(int)

        # McNemar's test (for paired predictions)
        # Count disagreements
        ml_right_rule_wrong = np.sum((y_pred_ml == y_true) & (y_pred_rule != y_true))
        ml_wrong_rule_right = np.sum((y_pred_ml != y_true) & (y_pred_rule == y_true))

        # Chi-square statistic
        if (ml_right_rule_wrong + ml_wrong_rule_right) > 0:
            mcnemar_statistic = ((abs(ml_right_rule_wrong - ml_wrong_rule_right) - 1) ** 2 /
                               (ml_right_rule_wrong + ml_wrong_rule_right))
        else:
            mcnemar_statistic = 0

        # Effect size (Cohen's h for proportions)
        p_ml = np.mean(ml_correct)
        p_rule = np.mean(rule_correct)
        cohens_h = 2 * (np.arcsin(np.sqrt(p_ml)) - np.arcsin(np.sqrt(p_rule)))

        return {
            'ml_accuracy': float(p_ml),
            'rule_accuracy': float(p_rule),
            'accuracy_difference': float(p_ml - p_rule),
            'accuracy_improvement_percentage': float((p_ml - p_rule) / p_rule * 100) if p_rule > 0 else 0,
            'mcnemar_statistic': float(mcnemar_statistic),
            'cohens_h': float(cohens_h),
            'ml_right_rule_wrong': int(ml_right_rule_wrong),
            'ml_wrong_rule_right': int(ml_wrong_rule_right),
            'both_right': int(np.sum(ml_correct & rule_correct)),
            'both_wrong': int(np.sum(~ml_correct & ~rule_correct))
        }

    def _per_class_comparison(self, y_true: np.ndarray, y_pred_ml: np.ndarray,
                              y_pred_rule: np.ndarray) -> dict:
        """
        Compare models on a per-class basis

        Args:
            y_true: True labels
            y_pred_ml: ML predictions
            y_pred_rule: Rule-based predictions

        Returns:
            Per-class comparison
        """
        comparison = {}

        for class_id, class_name in config.MOVEMENT_CLASSES.items():
            class_mask = y_true == class_id

            if np.sum(class_mask) == 0:
                continue

            y_true_class = y_true[class_mask]
            y_pred_ml_class = y_pred_ml[class_mask]
            y_pred_rule_class = y_pred_rule[class_mask]

            ml_accuracy = accuracy_score(y_true_class, y_pred_ml_class)
            rule_accuracy = accuracy_score(y_true_class, y_pred_rule_class)

            comparison[class_name] = {
                'sample_count': int(np.sum(class_mask)),
                'ml_accuracy': float(ml_accuracy),
                'rule_accuracy': float(rule_accuracy),
                'accuracy_difference': float(ml_accuracy - rule_accuracy),
                'ml_better': bool(ml_accuracy > rule_accuracy)
            }

        return comparison

    def save_results(self, output_dir: str = None):
        """
        Save test results

        Args:
            output_dir: Output directory
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(config.RESULTS_DIR, f'test_{timestamp}')

        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results
        json_path = os.path.join(output_dir, 'test_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {json_path}")

        # Generate visualizations
        self._plot_comparison(output_dir)

        # Generate text report
        self._generate_text_report(output_dir)

        return output_dir

    def _plot_comparison(self, output_dir: str):
        """
        Generate comparison visualizations

        Args:
            output_dir: Output directory
        """
        # 1. Accuracy comparison bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        models = ['ML Model', 'Rule-Based']
        accuracies = [
            self.results['ml_model']['accuracy'],
            self.results['rule_based_model']['accuracy']
        ]

        bars = ax.bar(models, accuracies, color=['#2ecc71', '#e74c3c'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Comparison: Overall Accuracy')
        ax.set_ylim([0, 1.0])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
        plt.close()

        # 2. Per-class F1 score comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        classes = list(config.MOVEMENT_CLASSES.values())
        ml_f1 = [self.results['ml_model']['per_class_metrics'][c]['f1_score'] for c in classes]
        rule_f1 = [self.results['rule_based_model']['per_class_metrics'][c]['f1_score'] for c in classes]

        x = np.arange(len(classes))
        width = 0.35

        ax.bar(x - width/2, ml_f1, width, label='ML Model', color='#3498db')
        ax.bar(x + width/2, rule_f1, width, label='Rule-Based', color='#e67e22')

        ax.set_xlabel('Class')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.0])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_f1_comparison.png'), dpi=300)
        plt.close()

        # 3. Confusion matrices (side by side)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ml_cm = np.array(self.results['ml_model']['confusion_matrix'])
        rule_cm = np.array(self.results['rule_based_model']['confusion_matrix'])

        class_labels = list(config.MOVEMENT_CLASSES.values())

        sns.heatmap(ml_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels,
                   ax=axes[0])
        axes[0].set_title('ML Model Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        sns.heatmap(rule_cm, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=class_labels, yticklabels=class_labels,
                   ax=axes[1])
        axes[1].set_title('Rule-Based Model Confusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
        plt.close()

        print("Visualizations saved")

    def _generate_text_report(self, output_dir: str):
        """
        Generate human-readable text report

        Args:
            output_dir: Output directory
        """
        report_path = os.path.join(output_dir, 'test_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("Comparison of ML-Based vs Rule-Based Compensatory Movement Detection\n")
            f.write("=" * 100 + "\n\n")

            # Test info
            f.write("TEST INFORMATION\n")
            f.write("-" * 100 + "\n")
            f.write(f"Test Set Size: {self.results['test_info']['test_size']} sequences\n")
            f.write(f"Number of Classes: {self.results['test_info']['num_classes']}\n")
            f.write("\nClass Distribution:\n")
            for class_name, count in self.results['test_info']['class_distribution'].items():
                f.write(f"  {class_name}: {count}\n")
            f.write("\n")

            # Overall comparison
            f.write("OVERALL PERFORMANCE COMPARISON\n")
            f.write("-" * 100 + "\n")
            f.write(f"ML Model Accuracy:         {self.results['ml_model']['accuracy']:.4f}\n")
            f.write(f"Rule-Based Model Accuracy: {self.results['rule_based_model']['accuracy']:.4f}\n")
            f.write(f"Accuracy Difference:       {self.results['statistical_comparison']['accuracy_difference']:.4f}\n")
            f.write(f"Improvement:               {self.results['statistical_comparison']['accuracy_improvement_percentage']:.2f}%\n")
            f.write("\n")

            f.write(f"ML Model F1 Score:         {self.results['ml_model']['f1_score']:.4f}\n")
            f.write(f"Rule-Based Model F1 Score: {self.results['rule_based_model']['f1_score']:.4f}\n")
            f.write("\n")

            # Statistical comparison
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-" * 100 + "\n")
            stat = self.results['statistical_comparison']
            f.write(f"ML correct, Rule wrong: {stat['ml_right_rule_wrong']}\n")
            f.write(f"ML wrong, Rule correct: {stat['ml_wrong_rule_right']}\n")
            f.write(f"Both correct: {stat['both_right']}\n")
            f.write(f"Both wrong: {stat['both_wrong']}\n")
            f.write(f"McNemar's statistic: {stat['mcnemar_statistic']:.4f}\n")
            f.write(f"Cohen's h (effect size): {stat['cohens_h']:.4f}\n")
            f.write("\n")

            # Per-class comparison
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 100 + "\n\n")

            for class_name in config.MOVEMENT_CLASSES.values():
                ml_metrics = self.results['ml_model']['per_class_metrics'][class_name]
                rule_metrics = self.results['rule_based_model']['per_class_metrics'][class_name]

                f.write(f"{class_name.upper()}\n")
                f.write(f"  Samples: {ml_metrics['support']}\n")
                f.write(f"  ML Model:\n")
                f.write(f"    Precision: {ml_metrics['precision']:.4f}\n")
                f.write(f"    Recall:    {ml_metrics['recall']:.4f}\n")
                f.write(f"    F1 Score:  {ml_metrics['f1_score']:.4f}\n")
                f.write(f"  Rule-Based Model:\n")
                f.write(f"    Precision: {rule_metrics['precision']:.4f}\n")
                f.write(f"    Recall:    {rule_metrics['recall']:.4f}\n")
                f.write(f"    F1 Score:  {rule_metrics['f1_score']:.4f}\n")
                f.write("\n")

            f.write("=" * 100 + "\n")

        print(f"Text report saved to: {report_path}")


def main():
    """
    Main testing script
    """
    import argparse

    parser = argparse.ArgumentParser(description='Test and compare models')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained ML model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (.npz file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from: {args.data}")
    data = np.load(args.data)

    if 'X_test' in data and 'y_test' in data:
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        # Use all data as test
        X_test = data['X']
        y_test = data['y']

    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Initialize tester
    tester = ModelTester(args.model)

    # Run tests
    results = tester.test_models(X_test, y_test)

    # Print summary
    print("\n" + "="*100)
    print("TEST RESULTS SUMMARY")
    print("="*100)
    print(f"ML Model Accuracy: {results['ml_model']['accuracy']:.4f}")
    print(f"Rule-Based Accuracy: {results['rule_based_model']['accuracy']:.4f}")
    print(f"Improvement: {results['statistical_comparison']['accuracy_improvement_percentage']:.2f}%")
    print("="*100)

    # Save results
    output_dir = tester.save_results(args.output)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
