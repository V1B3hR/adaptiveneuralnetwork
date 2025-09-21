"""
Bitext dataset loader for Adaptive Neural Network.

This module provides utilities for loading and preprocessing bitext datasets
for training language tasks. It supports loading from Kaggle via kagglehub
with graceful fallback to local CSV files.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BitextDatasetLoader:
    """
    Loader for bitext datasets with kagglehub integration and local fallback.

    This class provides a clean API for loading bitext datasets either from
    Kaggle (if credentials are available) or from local CSV files.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        local_path: Optional[Union[str, Path]] = None,
        use_kaggle: bool = True,
        text_column: str = "text",
        label_column: str = "label",
        sampling_fraction: Optional[float] = None,
        normalize_text: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize the bitext dataset loader.

        Args:
            dataset_name: Kaggle dataset name (e.g., "username/dataset-name")
            local_path: Path to local CSV file (fallback)
            use_kaggle: Whether to attempt Kaggle download
            text_column: Name of text column in dataset
            label_column: Name of label column in dataset
            sampling_fraction: Optional fraction to sample (0.0-1.0)
            normalize_text: Whether to apply basic text normalization
            random_seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.local_path = Path(local_path) if local_path else None
        self.use_kaggle = use_kaggle
        self.text_column = text_column
        self.label_column = label_column
        self.sampling_fraction = sampling_fraction
        self.normalize_text = normalize_text
        self.random_seed = random_seed

        # Check for optional dependencies
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if optional nlp dependencies are available."""
        self.has_pandas = self._try_import("pandas")
        self.has_kagglehub = self._try_import("kagglehub")
        self.has_sklearn = self._try_import("sklearn")

        if not self.has_pandas:
            warnings.warn(
                "pandas not available. Install with: pip install 'adaptiveneuralnetwork[nlp]'"
            )

    def _try_import(self, module_name: str) -> bool:
        """Try to import a module and return True if successful."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def _has_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are available."""
        return (
            os.environ.get("KAGGLE_USERNAME") is not None
            and os.environ.get("KAGGLE_KEY") is not None
        ) or Path.home().joinpath(".kaggle", "kaggle.json").exists()

    def _download_from_kaggle(self, dataset_name: str) -> Optional[Path]:
        """
        Download dataset from Kaggle using kagglehub.

        Returns:
            Path to downloaded dataset directory, or None if failed
        """
        if not self.has_kagglehub:
            logger.warning(
                "kagglehub not available. Install with: pip install 'adaptiveneuralnetwork[nlp]'"
            )
            return None

        if not self._has_kaggle_credentials():
            logger.warning(
                "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "environment variables or place kaggle.json in ~/.kaggle/"
            )
            return None

        try:
            import kagglehub

            logger.info(f"Downloading dataset from Kaggle: {dataset_name}")
            dataset_path = kagglehub.dataset_download(dataset_name)
            return Path(dataset_path)

        except Exception as e:
            logger.error(f"Failed to download dataset from Kaggle: {e}")
            return None

    def _load_csv(self, csv_path: Path) -> Optional["pandas.DataFrame"]:
        """
        Load CSV file using pandas.

        Returns:
            DataFrame or None if loading failed
        """
        if not self.has_pandas:
            logger.error("pandas required for CSV loading")
            return None

        try:
            import pandas as pd

            logger.info(f"Loading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)

            # Check if required columns exist
            if self.text_column not in df.columns:
                logger.error(f"Text column '{self.text_column}' not found in CSV")
                return None

            if self.label_column not in df.columns:
                logger.warning(f"Label column '{self.label_column}' not found in CSV")
                # For unsupervised tasks, this might be OK

            return df

        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            return None

    def _normalize_text(self, text: str) -> str:
        """Apply basic text normalization."""
        if not isinstance(text, str):
            return str(text)

        # Basic normalization
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace

        return text

    def _preprocess_dataframe(self, df: "pandas.DataFrame") -> "pandas.DataFrame":
        """
        Apply preprocessing to the dataframe.

        Args:
            df: Input dataframe

        Returns:
            Preprocessed dataframe
        """
        # Apply sampling if requested
        if self.sampling_fraction is not None and 0 < self.sampling_fraction < 1.0:
            logger.info(f"Sampling {self.sampling_fraction:.2%} of data")
            df = df.sample(frac=self.sampling_fraction, random_state=self.random_seed)

        # Apply text normalization if requested
        if self.normalize_text and self.text_column in df.columns:
            logger.info("Applying text normalization")
            df[self.text_column] = df[self.text_column].apply(self._normalize_text)

        # Remove rows with missing text
        df = df.dropna(subset=[self.text_column])

        logger.info(f"Preprocessed dataset size: {len(df)} samples")
        return df

    def _split_dataset(
        self, df: "pandas.DataFrame", val_split: float = 0.2
    ) -> Tuple["pandas.DataFrame", "pandas.DataFrame"]:
        """
        Split dataset into train and validation sets.

        Args:
            df: Input dataframe
            val_split: Fraction for validation set

        Returns:
            Tuple of (train_df, val_df)
        """
        if not self.has_sklearn:
            logger.warning("scikit-learn not available for stratified split, using random split")
            # Simple random split
            val_size = int(len(df) * val_split)
            df_shuffled = df.sample(frac=1.0, random_state=self.random_seed)
            val_df = df_shuffled.iloc[:val_size]
            train_df = df_shuffled.iloc[val_size:]
            return train_df, val_df

        try:
            from sklearn.model_selection import train_test_split

            # Use stratified split if labels are available
            if self.label_column in df.columns:
                train_df, val_df = train_test_split(
                    df,
                    test_size=val_split,
                    stratify=df[self.label_column],
                    random_state=self.random_seed,
                )
            else:
                train_df, val_df = train_test_split(
                    df, test_size=val_split, random_state=self.random_seed
                )

            return train_df, val_df

        except Exception as e:
            logger.warning(f"Stratified split failed, using random split: {e}")
            # Fallback to random split
            val_size = int(len(df) * val_split)
            df_shuffled = df.sample(frac=1.0, random_state=self.random_seed)
            val_df = df_shuffled.iloc[:val_size]
            train_df = df_shuffled.iloc[val_size:]
            return train_df, val_df

    def load_dataset(
        self, val_split: float = 0.2, force_local: bool = False
    ) -> Tuple[Optional["pandas.DataFrame"], Optional["pandas.DataFrame"]]:
        """
        Load the bitext dataset.

        Args:
            val_split: Fraction for validation set
            force_local: Force using local file instead of Kaggle

        Returns:
            Tuple of (train_df, val_df) or (None, None) if loading failed
        """
        df = None

        # Try Kaggle download first (if enabled and not forced local)
        if self.use_kaggle and not force_local and self.dataset_name:
            dataset_dir = self._download_from_kaggle(self.dataset_name)
            if dataset_dir:
                # Look for CSV files in the downloaded directory
                csv_files = list(dataset_dir.glob("*.csv"))
                if csv_files:
                    # Use the first CSV file found
                    df = self._load_csv(csv_files[0])

        # Fallback to local file
        if df is None and self.local_path and self.local_path.exists():
            df = self._load_csv(self.local_path)

        # If still no data, return None
        if df is None:
            logger.error("Failed to load dataset from any source")
            return None, None

        # Preprocess the data
        df = self._preprocess_dataframe(df)

        # Split into train/val
        train_df, val_df = self._split_dataset(df, val_split)

        logger.info("Dataset loaded successfully:")
        logger.info(f"  Training samples: {len(train_df)}")
        logger.info(f"  Validation samples: {len(val_df)}")

        return train_df, val_df

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset configuration.

        Returns:
            Dictionary with dataset information
        """
        return {
            "dataset_name": self.dataset_name,
            "local_path": str(self.local_path) if self.local_path else None,
            "use_kaggle": self.use_kaggle,
            "text_column": self.text_column,
            "label_column": self.label_column,
            "sampling_fraction": self.sampling_fraction,
            "normalize_text": self.normalize_text,
            "random_seed": self.random_seed,
            "has_pandas": self.has_pandas,
            "has_kagglehub": self.has_kagglehub,
            "has_sklearn": self.has_sklearn,
            "has_kaggle_credentials": self._has_kaggle_credentials(),
        }


def create_synthetic_bitext_data(
    num_samples: int = 1000,
    text_length_range: Tuple[int, int] = (10, 100),
    num_classes: int = 2,
    random_seed: int = 42,
) -> Tuple["pandas.DataFrame", "pandas.DataFrame"]:
    """
    Create synthetic bitext data for testing and demonstration.

    Args:
        num_samples: Number of samples to generate
        text_length_range: Range of text lengths (min, max)
        num_classes: Number of classes
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df)
    """
    try:
        import random

        import pandas as pd

        random.seed(random_seed)

        # Generate synthetic text data
        data = []
        words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "hello",
            "world",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "neural",
            "network",
            "adaptive",
            "system",
            "bitext",
            "dataset",
        ]

        for i in range(num_samples):
            # Generate random text
            text_length = random.randint(*text_length_range)
            text = " ".join(random.choices(words, k=text_length))

            # Generate random label
            label = random.randint(0, num_classes - 1)

            data.append({"text": text, "label": label, "id": i})

        df = pd.DataFrame(data)

        # Split into train/val (80/20)
        val_size = int(len(df) * 0.2)
        val_df = df.iloc[:val_size].copy()
        train_df = df.iloc[val_size:].copy()

        return train_df, val_df

    except ImportError:
        logger.error("pandas required for synthetic data generation")
        return None, None


# Example usage and demo
if __name__ == "__main__":
    # Demo the bitext dataset loader
    print("Bitext Dataset Loader Demo")
    print("=" * 40)

    # Create synthetic data for demo
    print("\n1. Creating synthetic bitext data...")
    train_df, val_df = create_synthetic_bitext_data(num_samples=100)

    if train_df is not None:
        print(f"   Training samples: {len(train_df)}")
        print(f"   Validation samples: {len(val_df)}")
        print(f"   Sample text: {train_df.iloc[0]['text'][:50]}...")

    # Demo loader configuration
    print("\n2. Configuring bitext dataset loader...")
    loader = BitextDatasetLoader(
        dataset_name="example/dataset", sampling_fraction=0.1, normalize_text=True
    )

    info = loader.get_dataset_info()
    print("   Configuration:")
    for key, value in info.items():
        print(f"     {key}: {value}")

    print("\n3. Demo complete!")
    print("   To use with real data:")
    print("   - Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
    print("   - Or provide local_path to CSV file")
    print("   - Install optional dependencies: pip install 'adaptiveneuralnetwork[nlp]'")
