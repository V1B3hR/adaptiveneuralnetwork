"""
Text classification baseline using scikit-learn TF-IDF + LogisticRegression.

This module provides a lightweight baseline for text classification tasks
that can be used to demonstrate state-modulated behavior without requiring
heavy GPU/transformer dependencies.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class TextClassificationBaseline:
    """
    Lightweight text classification baseline using TF-IDF and LogisticRegression.
    
    This baseline provides deterministic results and is suitable for smoke tests
    and demonstration of state-modulated behavior in the adaptive neural network.
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        Initialize the text classification baseline.
        
        Args:
            max_features: Maximum number of TF-IDF features
            min_df: Minimum document frequency for features
            max_df: Maximum document frequency for features
            ngram_range: N-gram range for TF-IDF
            C: Regularization parameter for LogisticRegression
            random_state: Random seed for reproducibility
            verbose: Whether to enable verbose logging
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.C = C
        self.random_state = random_state
        self.verbose = verbose
        
        # Model components (initialized in fit)
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_fitted = False
        
        # Training metrics
        self.training_history = []
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import sklearn
            self.has_sklearn = True
        except ImportError:
            self.has_sklearn = False
            warnings.warn("scikit-learn not available. Install with: pip install 'adaptiveneuralnetwork[nlp]'", stacklevel=2)
    
    def _create_vectorizer(self):
        """Create TF-IDF vectorizer."""
        if not self.has_sklearn:
            raise ImportError("scikit-learn required for TF-IDF vectorization")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        return TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
    
    def _create_classifier(self):
        """Create logistic regression classifier."""
        if not self.has_sklearn:
            raise ImportError("scikit-learn required for classification")
        
        from sklearn.linear_model import LogisticRegression
        
        return LogisticRegression(
            C=self.C,
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'
        )
    
    def _create_label_encoder(self):
        """Create label encoder for string labels."""
        if not self.has_sklearn:
            raise ImportError("scikit-learn required for label encoding")
        
        from sklearn.preprocessing import LabelEncoder
        return LabelEncoder()
    
    def fit(
        self, 
        texts: List[str], 
        labels: List[Union[str, int]],
        validation_texts: Optional[List[str]] = None,
        validation_labels: Optional[List[Union[str, int]]] = None
    ) -> Dict[str, Any]:
        """
        Fit the baseline model.
        
        Args:
            texts: Training texts
            labels: Training labels
            validation_texts: Optional validation texts
            validation_labels: Optional validation labels
            
        Returns:
            Dictionary with training metrics
        """
        if not self.has_sklearn:
            raise ImportError("scikit-learn required for model training")
        
        logger.info("Starting baseline model training...")
        
        # Create model components
        self.vectorizer = self._create_vectorizer()
        self.classifier = self._create_classifier()
        self.label_encoder = self._create_label_encoder()
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Vectorize texts
        if self.verbose:
            logger.info(f"Vectorizing {len(texts)} training texts...")
        
        X_train = self.vectorizer.fit_transform(texts)
        
        if self.verbose:
            logger.info(f"TF-IDF feature matrix shape: {X_train.shape}")
        
        # Train classifier
        if self.verbose:
            logger.info("Training logistic regression classifier...")
        
        self.classifier.fit(X_train, encoded_labels)
        
        # Calculate training metrics
        from sklearn.metrics import accuracy_score, classification_report
        
        train_predictions = self.classifier.predict(X_train)
        train_accuracy = accuracy_score(encoded_labels, train_predictions)
        
        metrics = {
            "train_accuracy": float(train_accuracy),
            "num_features": X_train.shape[1],
            "num_classes": len(self.label_encoder.classes_),
            "class_names": self.label_encoder.classes_.tolist()
        }
        
        # Validation metrics if provided
        if validation_texts and validation_labels:
            val_encoded_labels = self.label_encoder.transform(validation_labels)
            X_val = self.vectorizer.transform(validation_texts)
            val_predictions = self.classifier.predict(X_val)
            val_accuracy = accuracy_score(val_encoded_labels, val_predictions)
            
            metrics["val_accuracy"] = float(val_accuracy)
            
            if self.verbose:
                val_report = classification_report(
                    val_encoded_labels, 
                    val_predictions,
                    target_names=[str(c) for c in self.label_encoder.classes_],
                    output_dict=True
                )
                metrics["val_classification_report"] = val_report
        
        # Store training history
        self.training_history.append(metrics)
        self.is_fitted = True
        
        if self.verbose:
            logger.info(f"Training complete. Train accuracy: {train_accuracy:.4f}")
            if "val_accuracy" in metrics:
                logger.info(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
        
        return metrics
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self.vectorizer.transform(texts)
        encoded_predictions = self.classifier.predict(X)
        
        # Decode predictions back to original labels
        predictions = self.label_encoder.inverse_transform(encoded_predictions)
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self.vectorizer.transform(texts)
        probabilities = self.classifier.predict_proba(X)
        return probabilities
    
    def evaluate(
        self, 
        texts: List[str], 
        labels: List[Union[str, int]]
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a test set.
        
        Args:
            texts: Test texts
            labels: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        from sklearn.metrics import (
            accuracy_score, 
            precision_recall_fscore_support,
            classification_report,
            confusion_matrix
        )
        
        # Make predictions
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Get per-class metrics
        classification_rep = classification_report(
            labels, predictions, output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "num_samples": len(texts),
            "classification_report": classification_rep,
            "confusion_matrix": conf_matrix.tolist()
        }
        
        return metrics
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features for each class.
        
        Args:
            top_k: Number of top features to return per class
            
        Returns:
            Dictionary mapping class names to top features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.classifier.coef_
        
        feature_importance = {}
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            if len(self.label_encoder.classes_) == 2:
                # Binary classification - use single coefficient vector
                coef = coefficients[0] if i == 1 else -coefficients[0]
            else:
                # Multi-class classification
                coef = coefficients[i]
            
            # Get top positive and negative features
            top_indices = np.argsort(np.abs(coef))[-top_k:][::-1]
            top_features = [
                (feature_names[idx], float(coef[idx])) 
                for idx in top_indices
            ]
            
            feature_importance[str(class_name)] = top_features
        
        return feature_importance
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import pickle
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
            "config": {
                "max_features": self.max_features,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "ngram_range": self.ngram_range,
                "C": self.C,
                "random_state": self.random_state
            },
            "training_history": self.training_history
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        import pickle
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data["vectorizer"]
        self.classifier = model_data["classifier"]
        self.label_encoder = model_data["label_encoder"]
        self.training_history = model_data.get("training_history", [])
        
        # Update config from saved model
        config = model_data.get("config", {})
        for key, value in config.items():
            setattr(self, key, value)
        
        self.is_fitted = True
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "ngram_range": self.ngram_range,
            "C": self.C,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "has_sklearn": self.has_sklearn
        }
        
        if self.is_fitted:
            info.update({
                "num_features": self.vectorizer.transform([""]).shape[1],
                "num_classes": len(self.label_encoder.classes_),
                "class_names": self.label_encoder.classes_.tolist(),
                "training_epochs": len(self.training_history)
            })
        
        return info


def create_demo_baseline(
    num_samples: int = 1000,
    num_classes: int = 3,
    random_state: int = 42
) -> Tuple[TextClassificationBaseline, Dict[str, Any]]:
    """
    Create a demo baseline model with synthetic data.
    
    Args:
        num_samples: Number of synthetic samples
        num_classes: Number of classes  
        random_state: Random seed
        
    Returns:
        Tuple of (fitted_model, metrics)
    """
    try:
        import pandas as pd
        from adaptiveneuralnetwork.training.bitext_dataset import create_synthetic_bitext_data
        
        # Create synthetic data
        train_df, val_df = create_synthetic_bitext_data(
            num_samples=num_samples,
            num_classes=num_classes,
            random_seed=random_state
        )
        
        # Create and train baseline
        baseline = TextClassificationBaseline(
            max_features=1000,  # Smaller for demo
            random_state=random_state,
            verbose=True
        )
        
        metrics = baseline.fit(
            texts=train_df['text'].tolist(),
            labels=train_df['label'].tolist(),
            validation_texts=val_df['text'].tolist(),
            validation_labels=val_df['label'].tolist()
        )
        
        return baseline, metrics
        
    except ImportError as e:
        logger.error(f"Demo requires additional dependencies: {e}")
        return None, None


# Example usage and demo
if __name__ == "__main__":
    print("Text Classification Baseline Demo")
    print("=" * 40)
    
    # Create demo model
    print("\n1. Creating demo baseline with synthetic data...")
    baseline, metrics = create_demo_baseline(num_samples=500, num_classes=2)
    
    if baseline is not None:
        print(f"   Training accuracy: {metrics['train_accuracy']:.4f}")
        print(f"   Validation accuracy: {metrics['val_accuracy']:.4f}")
        print(f"   Number of features: {metrics['num_features']}")
        print(f"   Number of classes: {metrics['num_classes']}")
        
        # Show feature importance
        print("\n2. Top features per class:")
        feature_importance = baseline.get_feature_importance(top_k=5)
        for class_name, features in feature_importance.items():
            print(f"   Class {class_name}:")
            for feature, weight in features:
                print(f"     {feature}: {weight:.4f}")
        
        # Demo prediction
        print("\n3. Demo prediction:")
        test_texts = [
            "hello world artificial intelligence",
            "quick brown fox jumps over lazy dog"
        ]
        predictions = baseline.predict(test_texts)
        probabilities = baseline.predict_proba(test_texts)
        
        for text, pred, probs in zip(test_texts, predictions, probabilities, strict=False):
            print(f"   Text: '{text[:30]}...'")
            print(f"   Prediction: {pred}")
            print(f"   Probabilities: {[f'{p:.3f}' for p in probs]}")
    
    print("\n4. Demo complete!")
    print("   To use with real data:")
    print("   - Provide lists of texts and labels to fit() method")
    print("   - Install optional dependencies: pip install 'adaptiveneuralnetwork[nlp]'")