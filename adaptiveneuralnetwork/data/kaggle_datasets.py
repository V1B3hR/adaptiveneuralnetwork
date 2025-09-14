"""
Dataset loaders for Kaggle datasets specified in the problem statement:
1. ANNOMI Motivational Interviewing Dataset
2. Mental Health FAQs Dataset
"""

import csv
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import os

from ..benchmarks.text_classification import EssayDataset

logger = logging.getLogger(__name__)


def load_annomi_dataset(
    data_path: str,
    text_column: str = "text",
    label_column: str = "label",
    vocab_size: int = 10000,
    max_length: int = 512
) -> EssayDataset:
    """
    Load the ANNOMI Motivational Interviewing Dataset.
    
    Expected dataset URL: https://www.kaggle.com/datasets/rahulmenon1758/annomi-motivational-interviewing
    
    Args:
        data_path: Path to the dataset file (CSV, JSON, or directory)
        text_column: Name of the column containing text data
        label_column: Name of the column containing labels
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        
    Returns:
        EssayDataset instance
        
    Notes:
        This dataset likely contains motivational interviewing conversations.
        Labels might be binary (motivational/non-motivational) or multi-class
        (different types of motivational techniques).
    """
    logger.info(f"Loading ANNOMI dataset from {data_path}")
    
    texts, labels = _load_text_data(data_path, text_column, label_column)
    
    # Convert labels to binary if needed (for text classification)
    binary_labels = _convert_to_binary_labels(labels)
    
    logger.info(f"Loaded {len(texts)} samples from ANNOMI dataset")
    logger.info(f"Label distribution: {_get_label_distribution(binary_labels)}")
    
    return EssayDataset(
        texts=texts,
        labels=binary_labels,
        vocab_size=vocab_size,
        max_length=max_length
    )


def load_mental_health_faqs_dataset(
    data_path: str,
    question_column: str = "question",
    answer_column: str = "answer",
    category_column: str = "category",
    vocab_size: int = 10000,
    max_length: int = 512,
    use_questions_only: bool = False
) -> EssayDataset:
    """
    Load the Mental Health FAQs Dataset.
    
    Expected dataset URL: https://www.kaggle.com/datasets/ragishehab/mental-healthfaqs
    
    Args:
        data_path: Path to the dataset file (CSV, JSON, or directory)
        question_column: Name of the column containing questions
        answer_column: Name of the column containing answers
        category_column: Name of the column containing categories
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        use_questions_only: If True, only use questions; if False, concatenate Q&A
        
    Returns:
        EssayDataset instance
        
    Notes:
        This dataset contains mental health-related questions and answers.
        We can create a binary classification task (e.g., anxiety vs depression)
        or use the category column for multi-class classification.
    """
    logger.info(f"Loading Mental Health FAQs dataset from {data_path}")
    
    # Load the data - try different possible column names
    possible_question_cols = [question_column, "Question", "questions", "text", "query"]
    possible_answer_cols = [answer_column, "Answer", "answers", "response", "reply"]
    possible_category_cols = [category_column, "Category", "categories", "type", "class", "label"]
    
    data = _load_dataset_file(data_path)
    
    # Find the correct column names
    question_col = _find_column(data, possible_question_cols)
    answer_col = _find_column(data, possible_answer_cols)
    category_col = _find_column(data, possible_category_cols)
    
    texts = []
    labels = []
    
    for _, row in data.iterrows():
        if use_questions_only:
            text = str(row[question_col])
        else:
            # Concatenate question and answer
            question = str(row[question_col])
            answer = str(row[answer_col]) if answer_col else ""
            text = f"Q: {question} A: {answer}"
        
        texts.append(text)
        
        # Use category as label if available, otherwise create binary labels
        if category_col:
            labels.append(str(row[category_col]))
        else:
            # Default binary classification
            labels.append("mental_health")
    
    # Convert labels to binary
    binary_labels = _convert_to_binary_labels(labels)
    
    logger.info(f"Loaded {len(texts)} samples from Mental Health FAQs dataset")
    logger.info(f"Label distribution: {_get_label_distribution(binary_labels)}")
    
    return EssayDataset(
        texts=texts,
        labels=binary_labels,
        vocab_size=vocab_size,
        max_length=max_length
    )


def create_text_classification_dataset(
    texts: List[str],
    labels: List[str],
    vocab_size: int = 10000,
    max_length: int = 512
) -> EssayDataset:
    """
    Create a text classification dataset from raw texts and labels.
    
    Args:
        texts: List of text samples
        labels: List of string labels
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        
    Returns:
        EssayDataset instance
    """
    binary_labels = _convert_to_binary_labels(labels)
    
    return EssayDataset(
        texts=texts,
        labels=binary_labels,
        vocab_size=vocab_size,
        max_length=max_length
    )


def _load_text_data(
    data_path: str,
    text_column: str,
    label_column: str
) -> Tuple[List[str], List[str]]:
    """Load text data from various file formats."""
    data = _load_dataset_file(data_path)
    
    # Find the correct column names (case-insensitive)
    text_col = _find_column(data, [text_column, "text", "content", "message", "conversation"])
    label_col = _find_column(data, [label_column, "label", "category", "class", "type"])
    
    texts = [str(row[text_col]) for _, row in data.iterrows()]
    labels = [str(row[label_col]) for _, row in data.iterrows()]
    
    return texts, labels


def _load_dataset_file(data_path: str) -> pd.DataFrame:
    """Load dataset from file (CSV, JSON, or Excel)."""
    path = Path(data_path)
    
    if path.is_dir():
        # Look for common dataset files in directory
        for pattern in ["*.csv", "*.json", "*.xlsx", "*.tsv"]:
            files = list(path.glob(pattern))
            if files:
                data_path = str(files[0])
                logger.info(f"Found dataset file: {data_path}")
                break
        else:
            raise FileNotFoundError(f"No dataset files found in {data_path}")
    
    file_path = Path(data_path)
    
    if file_path.suffix.lower() == '.csv':
        try:
            return pd.read_csv(data_path)
        except:
            # Try with different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    return pd.read_csv(data_path, encoding=encoding)
                except:
                    continue
            raise
    elif file_path.suffix.lower() == '.json':
        # Handle both single JSON object and JSONL
        try:
            return pd.read_json(data_path)
        except:
            # Try JSONL format
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            return pd.DataFrame(data)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(data_path)
    elif file_path.suffix.lower() == '.tsv':
        return pd.read_csv(data_path, sep='\t')
    else:
        # Try to detect format
        try:
            return pd.read_csv(data_path)
        except:
            try:
                return pd.read_json(data_path)
            except:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _find_column(data: pd.DataFrame, possible_names: List[str]) -> str:
    """Find the correct column name from a list of possibilities."""
    columns = data.columns.tolist()
    
    # Exact match first
    for name in possible_names:
        if name in columns:
            return name
    
    # Case-insensitive match
    columns_lower = [col.lower() for col in columns]
    for name in possible_names:
        name_lower = name.lower()
        if name_lower in columns_lower:
            return columns[columns_lower.index(name_lower)]
    
    # Partial match
    for name in possible_names:
        name_lower = name.lower()
        for col in columns:
            if name_lower in col.lower():
                return col
    
    # If no match found, use first column
    logger.warning(f"Could not find column matching {possible_names}. Using first column: {columns[0]}")
    return columns[0]


def _convert_to_binary_labels(labels: List[str]) -> List[int]:
    """Convert string labels to binary labels (0, 1)."""
    unique_labels = list(set(labels))
    
    if len(unique_labels) == 2:
        # Already binary, just convert to 0/1
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    else:
        # Multi-class, convert to binary based on most common label
        label_counts = {label: labels.count(label) for label in unique_labels}
        most_common = max(label_counts, key=label_counts.get)
        
        # Most common becomes 0, everything else becomes 1
        label_map = {most_common: 0}
        for label in unique_labels:
            if label != most_common:
                label_map[label] = 1
    
    logger.info(f"Label mapping: {label_map}")
    return [label_map[label] for label in labels]


def _get_label_distribution(labels: List[int]) -> Dict[int, int]:
    """Get distribution of binary labels."""
    return {0: labels.count(0), 1: labels.count(1)}


# Dataset-specific helper functions
def get_dataset_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about supported datasets.
    
    Returns:
        Dictionary with dataset information
    """
    return {
        "annomi": {
            "name": "ANNOMI Motivational Interviewing Dataset",
            "url": "https://www.kaggle.com/datasets/rahulmenon1758/annomi-motivational-interviewing",
            "description": "Motivational interviewing conversations dataset",
            "loader": "load_annomi_dataset",
            "expected_columns": ["text", "label"],
            "task": "Binary classification (motivational vs non-motivational)"
        },
        "mental_health_faqs": {
            "name": "Mental Health FAQs Dataset", 
            "url": "https://www.kaggle.com/datasets/ragishehab/mental-healthfaqs",
            "description": "Mental health questions and answers dataset",
            "loader": "load_mental_health_faqs_dataset",
            "expected_columns": ["question", "answer", "category"],
            "task": "Binary classification (anxiety vs depression, etc.)"
        }
    }


def print_dataset_info():
    """Print information about supported datasets."""
    info = get_dataset_info()
    
    print("Supported Kaggle Datasets:")
    print("=" * 50)
    
    for dataset_key, dataset_info in info.items():
        print(f"\n{dataset_info['name']}")
        print(f"URL: {dataset_info['url']}")
        print(f"Description: {dataset_info['description']}")
        print(f"Loader function: {dataset_info['loader']}")
        print(f"Expected columns: {dataset_info['expected_columns']}")
        print(f"Task: {dataset_info['task']}")
        print("-" * 30)
    
    print("\nUsage Instructions:")
    print("1. Download the dataset from Kaggle")
    print("2. Extract to a local directory")
    print("3. Use the appropriate loader function")
    print("4. Run training with --data-path pointing to the dataset")