"""
Scikit-learn baseline model for intent classification.

This module provides a lightweight baseline using TF-IDF + LogisticRegression.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingResults:
    """Results from baseline model training."""
    train_accuracy: float
    test_accuracy: float
    training_time: float
    num_samples: int
    num_classes: int
    model_type: str
    dataset_type: str


class SklearnBaseline:
    """Scikit-learn baseline model for intent classification."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the baseline model.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.vectorizer = None
        self.classifier = None
        self.class_names = None
        self.is_trained = False
    
    def _ensure_sklearn_available(self):
        """Ensure scikit-learn is available."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for the baseline model. "
                "Install with: pip install 'adaptiveneuralnetwork[nlp]'"
            ) from e
    
    def train(self, texts: List[str], labels: List[int], class_names: List[str], 
              test_size: float = 0.2) -> TrainingResults:
        """
        Train the baseline model.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            class_names: List of class names
            test_size: Fraction of data to use for testing
            
        Returns:
            TrainingResults with performance metrics
        """
        self._ensure_sklearn_available()
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        logger.info(f"Training baseline model on {len(texts)} samples with {len(class_names)} classes")
        
        start_time = time.time()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=self.random_state,
            stratify=labels
        )
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit vectorizer and transform texts
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train logistic regression
        self.classifier = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'  # Better for multiclass, no deprecation warning
        )
        
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate
        train_pred = self.classifier.predict(X_train_tfidf)
        test_pred = self.classifier.predict(X_test_tfidf)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        training_time = time.time() - start_time
        
        self.class_names = class_names
        self.is_trained = True
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Train accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        return TrainingResults(
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            training_time=training_time,
            num_samples=len(texts),
            num_classes=len(class_names),
            model_type="TF-IDF + LogisticRegression",
            dataset_type="bitext"
        )
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict labels for texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of predicted label IDs
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_tfidf = self.vectorizer.transform(texts)
        return self.classifier.predict(X_tfidf).tolist()
    
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """
        Predict class probabilities for texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of probability arrays
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_tfidf = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X_tfidf).tolist()


def save_results_to_json(results: TrainingResults, output_path: str) -> None:
    """
    Save training results to JSON file.
    
    Args:
        results: TrainingResults to save
        output_path: Path to output JSON file
    """
    results_dict = {
        "train_accuracy": results.train_accuracy,
        "test_accuracy": results.test_accuracy,
        "training_time_seconds": results.training_time,
        "num_samples": results.num_samples,
        "num_classes": results.num_classes,
        "model_type": results.model_type,
        "dataset_type": results.dataset_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def run_baseline_training(dataset, mode: str = "smoke", output_dir: str = "results") -> TrainingResults:
    """
    Run baseline training on a dataset.
    
    Args:
        dataset: BitextDataset instance
        mode: Training mode ("smoke" for fast, "benchmark" for full)
        output_dir: Directory to save results
        
    Returns:
        TrainingResults
    """
    # Adjust dataset size based on mode
    if mode == "smoke":
        # Use smaller subset for smoke test
        num_samples = min(200, len(dataset))
        logger.info(f"Smoke mode: using {num_samples} samples")
        samples = dataset.samples[:num_samples]
        texts = [s.text for s in samples]
        labels = [s.intent_id for s in samples]
    else:
        # Use full dataset for benchmark
        texts, labels = dataset.get_texts_and_labels()
        logger.info(f"Benchmark mode: using all {len(texts)} samples")
    
    # Train baseline model
    baseline = SklearnBaseline()
    results = baseline.train(texts, labels, dataset.class_names)
    
    # Save results
    output_path = Path(output_dir) / "metrics.json"
    save_results_to_json(results, str(output_path))
    
    return results