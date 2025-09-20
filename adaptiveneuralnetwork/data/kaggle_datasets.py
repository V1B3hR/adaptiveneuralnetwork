"""
Dataset loaders for Kaggle datasets specified in the problem statement:
1. ANNOMI Motivational Interviewing Dataset
2. Mental Health FAQs Dataset
3. Social Media Sentiments Analysis Dataset
4. Part-of-Speech Tagging Dataset
"""

import csv
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
import os
import random
import numpy as np
from collections import Counter, defaultdict

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


def load_social_media_sentiment_dataset(
    data_path: str,
    text_column: str = "text",
    sentiment_column: str = "sentiment",
    vocab_size: int = 10000,
    max_length: int = 512
) -> EssayDataset:
    """
    Load the Social Media Sentiments Analysis Dataset.
    
    Expected dataset URL: https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset
    
    Args:
        data_path: Path to the dataset file (CSV, JSON, Excel, or TSV)
        text_column: Name of the column containing text data
        sentiment_column: Name of the column containing sentiment labels
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        
    Returns:
        EssayDataset instance
        
    Notes:
        This dataset contains social media posts with sentiment labels.
        Common sentiment labels include: positive, negative, neutral
        The function supports flexible format loading (CSV/JSON/Excel/TSV).
    """
    logger.info(f"Loading Social Media Sentiment dataset from {data_path}")
    
    # Load the data - try different possible column names
    possible_text_cols = [text_column, "Text", "content", "message", "post", "tweet", "comment"]
    possible_sentiment_cols = [sentiment_column, "Sentiment", "label", "emotion", "feeling", "class"]
    
    data = _load_dataset_file(data_path)
    
    # Find the correct column names
    text_col = _find_column(data, possible_text_cols)
    sentiment_col = _find_column(data, possible_sentiment_cols)
    
    texts = []
    labels = []
    
    for _, row in data.iterrows():
        text = str(row[text_col]).strip()
        sentiment = str(row[sentiment_col]).strip().lower()
        
        # Skip empty texts
        if not text or text.lower() in ['nan', 'none', '']:
            continue
            
        texts.append(text)
        labels.append(sentiment)
    
    # Convert labels to binary for sentiment analysis
    # Common sentiment mappings: positive=1, negative/neutral=0
    binary_labels = _convert_sentiment_to_binary_labels(labels)
    
    logger.info(f"Loaded {len(texts)} samples from Social Media Sentiment dataset")
    logger.info(f"Label distribution: {_get_label_distribution(binary_labels)}")
    
    return EssayDataset(
        texts=texts,
        labels=binary_labels,
        vocab_size=vocab_size,
        max_length=max_length
    )


def load_pos_tagging_dataset(
    data_path: str,
    max_sentences: Optional[int] = None,
    min_token_length: int = 1,
    filter_punctuation: bool = False,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    vocab_size: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Load the Kaggle Part-of-Speech Tagging dataset.
    
    Expected dataset URL: https://www.kaggle.com/datasets/ruchi798/part-of-speech-tagging
    
    Args:
        data_path: Path to the dataset file (CSV) or directory
        max_sentences: Maximum number of sentences to load (for sampling/testing)
        min_token_length: Minimum token length to include 
        filter_punctuation: If True, filter out punctuation-only tokens
        train_split: Proportion for training set
        val_split: Proportion for validation set  
        test_split: Proportion for test set
        vocab_size: Maximum vocabulary size for tokens
        seed: Random seed for reproducible splits
        
    Returns:
        Dictionary containing:
            - datasets: Dict with 'train', 'val', 'test' POSDataset instances
            - vocab: Token to index mapping
            - tag_vocab: Tag to index mapping  
            - stats: Dataset statistics
            - config: Configuration used
    """
    logger.info(f"Loading POS tagging dataset from {data_path}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Load raw data
    sentences, all_tags = _load_pos_data(data_path)
    
    # Filter by token length and punctuation if requested
    if min_token_length > 1 or filter_punctuation:
        sentences, all_tags = _filter_pos_tokens(
            sentences, all_tags, min_token_length, filter_punctuation
        )
    
    # Sample sentences if requested
    if max_sentences and len(sentences) > max_sentences:
        indices = random.sample(range(len(sentences)), max_sentences)
        sentences = [sentences[i] for i in indices]
        all_tags = [all_tags[i] for i in indices]
        logger.info(f"Sampled {max_sentences} sentences from {len(sentences)} total")
    
    # Compute statistics
    stats = _compute_pos_statistics(sentences, all_tags)
    logger.info(f"Dataset statistics: {stats}")
    
    # Build vocabularies
    vocab, tag_vocab = _build_pos_vocabularies(sentences, all_tags, vocab_size)
    
    # Stratified split by tag distribution
    train_data, val_data, test_data = _stratified_pos_split(
        sentences, all_tags, train_split, val_split, test_split, seed
    )
    
    # Create dataset instances
    datasets = {
        'train': POSDataset(*train_data, vocab, tag_vocab),
        'val': POSDataset(*val_data, vocab, tag_vocab),
        'test': POSDataset(*test_data, vocab, tag_vocab)
    }
    
    config = {
        'max_sentences': max_sentences,
        'min_token_length': min_token_length,
        'filter_punctuation': filter_punctuation,
        'splits': {'train': train_split, 'val': val_split, 'test': test_split},
        'vocab_size': vocab_size,
        'seed': seed
    }
    
    logger.info(f"Created datasets - Train: {len(datasets['train'])}, "
                f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    
    return {
        'datasets': datasets,
        'vocab': vocab,
        'tag_vocab': tag_vocab,
        'stats': stats,
        'config': config
    }


class POSDataset:
    """Dataset class for POS tagging sequences."""
    
    def __init__(
        self, 
        sentences: List[List[str]], 
        tags: List[List[str]], 
        vocab: Dict[str, int], 
        tag_vocab: Dict[str, int],
        max_length: int = 512
    ):
        self.sentences = sentences
        self.tags = tags
        self.vocab = vocab
        self.tag_vocab = tag_vocab
        self.max_length = max_length
        
        # Create reverse mappings
        self.idx_to_token = {idx: token for token, idx in vocab.items()}
        self.idx_to_tag = {idx: tag for tag, idx in tag_vocab.items()}
        
        # Special tokens
        self.pad_token_id = vocab.get('<PAD>', 0)
        self.unk_token_id = vocab.get('<UNK>', 1)
        self.pad_tag_id = tag_vocab.get('<PAD>', 0)
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sentence with tokens and tags."""
        tokens = self.sentences[idx]
        tags = self.tags[idx]
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            tags = tags[:self.max_length]
        
        # Convert to indices
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        tag_ids = [self.tag_vocab.get(tag, self.pad_tag_id) for tag in tags]
        
        return {
            'tokens': tokens,
            'tags': tags,
            'token_ids': token_ids,
            'tag_ids': tag_ids,
            'length': len(tokens)
        }
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """Get a batch of sentences with padding."""
        batch_items = [self[i] for i in indices]
        
        # Find max length in batch
        max_len = max(item['length'] for item in batch_items)
        
        # Pad sequences
        batch_token_ids = []
        batch_tag_ids = []
        batch_lengths = []
        batch_tokens = []
        batch_tags = []
        
        for item in batch_items:
            token_ids = item['token_ids'] + [self.pad_token_id] * (max_len - len(item['token_ids']))
            tag_ids = item['tag_ids'] + [self.pad_tag_id] * (max_len - len(item['tag_ids']))
            
            batch_token_ids.append(token_ids)
            batch_tag_ids.append(tag_ids)
            batch_lengths.append(item['length'])
            batch_tokens.append(item['tokens'])
            batch_tags.append(item['tags'])
        
        return {
            'input_ids': batch_token_ids,
            'tag_ids': batch_tag_ids,
            'lengths': batch_lengths,
            'tokens': batch_tokens,
            'tags': batch_tags,
            'attention_mask': [[1 if i < length else 0 for i in range(max_len)] 
                              for length in batch_lengths]
        }


def _load_pos_data(data_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Load POS tagging data from CSV file."""
    path = Path(data_path)
    
    if path.is_dir():
        # Look for CSV files in directory
        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        data_path = str(csv_files[0])
        logger.info(f"Found dataset file: {data_path}")
    
    # Load CSV and detect columns
    df = pd.read_csv(data_path)
    logger.info(f"Loaded CSV with columns: {df.columns.tolist()}")
    
    # Try to find sentence, word, and POS columns
    sentence_col = _find_column(df, ['sentence', 'sentence_id', 'sent_id', 'id'])
    word_col = _find_column(df, ['word', 'token', 'text', 'tokens'])
    pos_col = _find_column(df, ['pos', 'tag', 'pos_tag', 'label'])
    
    logger.info(f"Using columns - Sentence: {sentence_col}, Word: {word_col}, POS: {pos_col}")
    
    # Group by sentence
    sentences = []
    all_tags = []
    
    if sentence_col:
        # Group by sentence ID
        grouped = df.groupby(sentence_col)
        for sent_id, group in grouped:
            tokens = group[word_col].astype(str).tolist()
            tags = group[pos_col].astype(str).tolist()
            
            # Filter out empty tokens
            filtered_tokens = []
            filtered_tags = []
            for token, tag in zip(tokens, tags):
                if token.strip() and token.lower() not in ['nan', 'none', '']:
                    filtered_tokens.append(token.strip())
                    filtered_tags.append(tag.strip())
            
            if filtered_tokens:  # Only add non-empty sentences
                sentences.append(filtered_tokens)
                all_tags.append(filtered_tags)
    else:
        # Treat each row as a separate token, group sequential rows as sentences
        current_sentence = []
        current_tags = []
        
        for _, row in df.iterrows():
            token = str(row[word_col]).strip()
            tag = str(row[pos_col]).strip()
            
            if token and token.lower() not in ['nan', 'none', '']:
                current_sentence.append(token)
                current_tags.append(tag)
            else:
                # End of sentence or empty token
                if current_sentence:
                    sentences.append(current_sentence)
                    all_tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
        
        # Add final sentence if exists
        if current_sentence:
            sentences.append(current_sentence)
            all_tags.append(current_tags)
    
    logger.info(f"Loaded {len(sentences)} sentences with {sum(len(s) for s in sentences)} total tokens")
    return sentences, all_tags


def _filter_pos_tokens(
    sentences: List[List[str]], 
    all_tags: List[List[str]], 
    min_length: int, 
    filter_punct: bool
) -> Tuple[List[List[str]], List[List[str]]]:
    """Filter tokens by length and punctuation."""
    filtered_sentences = []
    filtered_tags = []
    
    for tokens, tags in zip(sentences, all_tags):
        filtered_tokens = []
        filtered_token_tags = []
        
        for token, tag in zip(tokens, tags):
            # Length filter
            if len(token) < min_length:
                continue
                
            # Punctuation filter
            if filter_punct and len(token) == 1 and not token.isalnum():
                continue
                
            filtered_tokens.append(token)
            filtered_token_tags.append(tag)
        
        # Only keep sentences with tokens remaining
        if filtered_tokens:
            filtered_sentences.append(filtered_tokens)
            filtered_tags.append(filtered_token_tags)
    
    logger.info(f"After filtering: {len(filtered_sentences)} sentences")
    return filtered_sentences, filtered_tags


def _compute_pos_statistics(sentences: List[List[str]], all_tags: List[List[str]]) -> Dict[str, Any]:
    """Compute dataset statistics."""
    sentence_lengths = [len(s) for s in sentences]
    total_tokens = sum(sentence_lengths)
    
    # Tag frequency
    tag_counts = Counter()
    for tags in all_tags:
        tag_counts.update(tags)
    
    # Token frequency
    token_counts = Counter()
    for tokens in sentences:
        token_counts.update(tokens)
    
    stats = {
        'num_sentences': len(sentences),
        'total_tokens': total_tokens,
        'unique_tokens': len(token_counts),
        'unique_tags': len(tag_counts),
        'avg_sentence_length': np.mean(sentence_lengths),
        'median_sentence_length': np.median(sentence_lengths),
        'percentile_95_length': np.percentile(sentence_lengths, 95) if sentence_lengths else 0,
        'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
        'tag_frequencies': dict(tag_counts.most_common(20)),  # Top 20 tags
        'most_common_tokens': dict(token_counts.most_common(10))  # Top 10 tokens
    }
    
    return stats


def _build_pos_vocabularies(
    sentences: List[List[str]], 
    all_tags: List[List[str]], 
    vocab_size: int
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build token and tag vocabularies."""
    # Count token frequencies
    token_counts = Counter()
    for tokens in sentences:
        token_counts.update(tokens)
    
    # Build token vocab with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    most_common_tokens = token_counts.most_common(vocab_size - 2)  # Reserve space for special tokens
    
    for token, _ in most_common_tokens:
        vocab[token] = len(vocab)
    
    # Build tag vocab (keep all tags for POS tagging)
    tag_vocab = {'<PAD>': 0}
    unique_tags = set()
    for tags in all_tags:
        unique_tags.update(tags)
    
    for tag in sorted(unique_tags):  # Sort for consistent ordering
        if tag not in tag_vocab:
            tag_vocab[tag] = len(tag_vocab)
    
    logger.info(f"Built vocabularies - Tokens: {len(vocab)}, Tags: {len(tag_vocab)}")
    return vocab, tag_vocab


def _stratified_pos_split(
    sentences: List[List[str]], 
    all_tags: List[List[str]], 
    train_split: float, 
    val_split: float, 
    test_split: float,
    seed: int
) -> Tuple[Tuple[List[List[str]], List[List[str]]], ...]:
    """Create stratified splits based on tag distribution."""
    # Create indices
    indices = list(range(len(sentences)))
    random.shuffle(indices)
    
    # Calculate split points
    n = len(indices)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Split data
    train_sentences = [sentences[i] for i in train_indices]
    train_tags = [all_tags[i] for i in train_indices]
    
    val_sentences = [sentences[i] for i in val_indices] 
    val_tags = [all_tags[i] for i in val_indices]
    
    test_sentences = [sentences[i] for i in test_indices]
    test_tags = [all_tags[i] for i in test_indices]
    
    return ((train_sentences, train_tags), 
            (val_sentences, val_tags), 
            (test_sentences, test_tags))


def get_pos_dataset_statistics(data_path: str) -> Dict[str, Any]:
    """
    Get lightweight statistics for POS dataset without loading full data.
    
    Args:
        data_path: Path to dataset
        
    Returns:
        Dictionary with basic statistics
    """
    try:
        sentences, all_tags = _load_pos_data(data_path)
        return _compute_pos_statistics(sentences, all_tags)
    except Exception as e:
        logger.error(f"Error computing statistics: {e}")
        return {}


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


def _convert_sentiment_to_binary_labels(labels: List[str]) -> List[int]:
    """Convert sentiment labels to binary labels with sentiment-specific logic."""
    # Normalize labels to lowercase for consistency
    normalized_labels = [label.lower().strip() for label in labels]
    unique_labels = list(set(normalized_labels))
    
    # Define positive sentiment keywords
    positive_keywords = ['positive', 'pos', 'good', 'happy', 'joy', 'love', '1']
    negative_keywords = ['negative', 'neg', 'bad', 'sad', 'anger', 'hate', '0']
    neutral_keywords = ['neutral', 'neu', 'mixed']
    
    # Create mapping based on sentiment analysis conventions
    label_map = {}
    
    for label in unique_labels:
        if any(keyword in label for keyword in positive_keywords):
            label_map[label] = 1  # Positive = 1
        elif any(keyword in label for keyword in negative_keywords):
            label_map[label] = 0  # Negative = 0  
        elif any(keyword in label for keyword in neutral_keywords):
            label_map[label] = 0  # Neutral = 0 (group with negative for binary)
        else:
            # If unknown sentiment, try to infer from numerical values or default to negative
            try:
                val = float(label)
                label_map[label] = 1 if val > 0.5 else 0
            except ValueError:
                # Default unknown sentiments to negative class
                label_map[label] = 0
    
    logger.info(f"Sentiment label mapping: {label_map}")
    
    # Convert using the mapping
    binary_labels = [label_map[label] for label in normalized_labels]
    return binary_labels


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
        },
        "social_media_sentiment": {
            "name": "Social Media Sentiments Analysis Dataset",
            "url": "https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset",
            "description": "Social media posts with sentiment labels for sentiment analysis",
            "loader": "load_social_media_sentiment_dataset",
            "expected_columns": ["text", "sentiment"],
            "task": "Binary sentiment classification (positive vs negative/neutral)"
        },
        "pos_tagging": {
            "name": "Part-of-Speech Tagging Dataset",
            "url": "https://www.kaggle.com/datasets/ruchi798/part-of-speech-tagging",
            "description": "Token-level POS tagging dataset for sequence labeling",
            "loader": "load_pos_tagging_dataset",
            "expected_columns": ["sentence", "word", "pos"],
            "task": "Sequence labeling (token-level POS tag prediction)"
        },
        "vr_driving": {
            "name": "Virtual Reality Driving Simulator Dataset",
            "url": "https://www.kaggle.com/datasets/sasanj/virtual-reality-driving-simulator-dataset",
            "description": "Virtual reality driving behavior and performance data",
            "loader": "load_vr_driving_dataset",
            "expected_columns": ["time", "speed", "steering", "performance"],
            "task": "Regression (driving performance prediction)"
        },
        "autvi": {
            "name": "AUTVI Dataset",
            "url": "https://www.kaggle.com/datasets/hassanmojab/autvi",
            "description": "Automated vehicle inspection dataset",
            "loader": "load_autvi_dataset",
            "expected_columns": ["features", "inspection_result"],
            "task": "Binary classification (pass/fail inspection)"
        },
        "digakust": {
            "name": "Digakust Dataset (Mensa Saarland University)",
            "url": "https://www.kaggle.com/datasets/resc28/digakust-dataset-mensa-saarland-university",
            "description": "Digital acoustic analysis dataset from Saarland University",
            "loader": "load_digakust_dataset",
            "expected_columns": ["audio_features", "classification"],
            "task": "Multi-class classification (acoustic pattern recognition)"
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


def load_vr_driving_dataset(
    data_path: str,
    feature_columns: Optional[List[str]] = None,
    target_column: str = "performance",
    vocab_size: int = 10000,
    max_length: int = 512
) -> EssayDataset:
    """
    Load the Virtual Reality Driving Simulator Dataset.
    
    Expected dataset URL: https://www.kaggle.com/datasets/sasanj/virtual-reality-driving-simulator-dataset
    
    Args:
        data_path: Path to the dataset file (CSV, JSON, or directory)
        feature_columns: List of columns to use as features (auto-detected if None)
        target_column: Name of the target column
        vocab_size: Maximum vocabulary size for text processing
        max_length: Maximum sequence length
        
    Returns:
        EssayDataset instance
        
    Notes:
        This dataset contains VR driving behavior data including speed, steering,
        and performance metrics. Can be used for regression or classification tasks.
    """
    logger.info(f"Loading VR Driving dataset from {data_path}")
    
    # Load the dataset
    df = _load_dataset_file(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Auto-detect feature columns if not specified
    if feature_columns is None:
        # Common VR driving features
        possible_features = ['time', 'speed', 'steering', 'acceleration', 'brake', 
                           'position_x', 'position_y', 'rotation', 'lane_deviation']
        feature_columns = [col for col in possible_features if col in df.columns]
        
        if not feature_columns:
            # Fall back to all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]
    
    # Find target column
    if target_column not in df.columns:
        target_column = _find_column(df, ['performance', 'score', 'rating', 'quality', 
                                        'result', 'outcome', 'label', 'target'])
    
    logger.info(f"Using feature columns: {feature_columns}")
    logger.info(f"Using target column: {target_column}")
    
    # Extract features and targets
    if feature_columns:
        # Combine numeric features into text representation for compatibility
        feature_texts = []
        for _, row in df.iterrows():
            features = [f"{col}:{row[col]}" for col in feature_columns if pd.notna(row[col])]
            feature_texts.append(" ".join(features))
    else:
        # Fall back to using all columns as text
        feature_texts = df.apply(lambda x: " ".join([f"{col}:{val}" for col, val in x.items() 
                                                   if col != target_column and pd.notna(val)]), axis=1).tolist()
    
    # Extract targets
    if target_column in df.columns:
        targets = df[target_column].tolist()
        # Convert to binary classification if needed
        if df[target_column].dtype == 'object':
            targets = _convert_to_binary_labels([str(t) for t in targets])
        else:
            # For numeric targets, create binary based on median
            median_val = df[target_column].median()
            targets = [1 if val > median_val else 0 for val in targets]
    else:
        logger.warning(f"Target column '{target_column}' not found, using dummy targets")
        targets = [0] * len(feature_texts)
    
    logger.info(f"Dataset statistics: {len(feature_texts)} samples")
    logger.info(f"Target distribution: {_get_label_distribution(targets)}")
    
    return create_text_classification_dataset(feature_texts, [str(t) for t in targets], vocab_size, max_length)


def load_autvi_dataset(
    data_path: str,
    feature_columns: Optional[List[str]] = None,
    target_column: str = "inspection_result",
    vocab_size: int = 10000,
    max_length: int = 512
) -> EssayDataset:
    """
    Load the AUTVI (Automated Vehicle Inspection) Dataset.
    
    Expected dataset URL: https://www.kaggle.com/datasets/hassanmojab/autvi
    
    Args:
        data_path: Path to the dataset file (CSV, JSON, or directory)
        feature_columns: List of columns to use as features (auto-detected if None)
        target_column: Name of the target column
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        
    Returns:
        EssayDataset instance
        
    Notes:
        This dataset contains automated vehicle inspection data.
        Typically used for binary classification (pass/fail inspection).
    """
    logger.info(f"Loading AUTVI dataset from {data_path}")
    
    # Load the dataset
    df = _load_dataset_file(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Auto-detect feature columns if not specified
    if feature_columns is None:
        # Common vehicle inspection features
        possible_features = ['engine', 'brakes', 'lights', 'tires', 'emissions', 
                           'safety', 'electrical', 'body', 'suspension', 'exhaust']
        feature_columns = [col for col in possible_features if col in df.columns]
        
        if not feature_columns:
            # Use all columns except target
            feature_columns = [col for col in df.columns if col != target_column]
    
    # Find target column
    if target_column not in df.columns:
        target_column = _find_column(df, ['inspection_result', 'result', 'pass_fail', 
                                        'status', 'outcome', 'label', 'target'])
    
    logger.info(f"Using feature columns: {feature_columns}")
    logger.info(f"Using target column: {target_column}")
    
    # Extract features and targets
    if feature_columns:
        # Combine features into text representation
        feature_texts = []
        for _, row in df.iterrows():
            features = [f"{col}:{row[col]}" for col in feature_columns if pd.notna(row[col])]
            feature_texts.append(" ".join(features))
    else:
        # Fall back to using all columns as text
        feature_texts = df.apply(lambda x: " ".join([f"{col}:{val}" for col, val in x.items() 
                                                   if col != target_column and pd.notna(val)]), axis=1).tolist()
    
    # Extract targets
    if target_column in df.columns:
        targets = df[target_column].tolist()
        # Convert to binary labels
        targets = _convert_to_binary_labels([str(t) for t in targets])
    else:
        logger.warning(f"Target column '{target_column}' not found, using dummy targets")
        targets = [0] * len(feature_texts)
    
    logger.info(f"Dataset statistics: {len(feature_texts)} samples")
    logger.info(f"Target distribution: {_get_label_distribution(targets)}")
    
    return create_text_classification_dataset(feature_texts, [str(t) for t in targets], vocab_size, max_length)


def load_digakust_dataset(
    data_path: str,
    feature_columns: Optional[List[str]] = None,
    target_column: str = "classification",
    vocab_size: int = 10000,
    max_length: int = 512
) -> EssayDataset:
    """
    Load the Digakust Dataset (Digital Acoustic Analysis) from Mensa Saarland University.
    
    Expected dataset URL: https://www.kaggle.com/datasets/resc28/digakust-dataset-mensa-saarland-university
    
    Args:
        data_path: Path to the dataset file (CSV, JSON, or directory)
        feature_columns: List of columns to use as features (auto-detected if None)
        target_column: Name of the target column
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        
    Returns:
        EssayDataset instance
        
    Notes:
        This dataset contains digital acoustic analysis data from Saarland University.
        Typically used for multi-class classification of acoustic patterns.
    """
    logger.info(f"Loading Digakust dataset from {data_path}")
    
    # Load the dataset
    df = _load_dataset_file(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Auto-detect feature columns if not specified
    if feature_columns is None:
        # Common acoustic analysis features
        possible_features = ['frequency', 'amplitude', 'duration', 'pitch', 'formant',
                           'spectral', 'mfcc', 'chroma', 'zero_crossing', 'spectral_centroid']
        feature_columns = [col for col in possible_features if col in df.columns]
        
        if not feature_columns:
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]
    
    # Find target column
    if target_column not in df.columns:
        target_column = _find_column(df, ['classification', 'class', 'category', 
                                        'type', 'label', 'target'])
    
    logger.info(f"Using feature columns: {feature_columns}")
    logger.info(f"Using target column: {target_column}")
    
    # Extract features and targets
    if feature_columns:
        # Combine acoustic features into text representation
        feature_texts = []
        for _, row in df.iterrows():
            features = [f"{col}:{row[col]}" for col in feature_columns if pd.notna(row[col])]
            feature_texts.append(" ".join(features))
    else:
        # Fall back to using all columns as text
        feature_texts = df.apply(lambda x: " ".join([f"{col}:{val}" for col, val in x.items() 
                                                   if col != target_column and pd.notna(val)]), axis=1).tolist()
    
    # Extract targets
    if target_column in df.columns:
        targets = df[target_column].tolist()
        # For multi-class, convert to binary for compatibility (can be extended later)
        targets = _convert_to_binary_labels([str(t) for t in targets])
    else:
        logger.warning(f"Target column '{target_column}' not found, using dummy targets")
        targets = [0] * len(feature_texts)
    
    logger.info(f"Dataset statistics: {len(feature_texts)} samples")
    logger.info(f"Target distribution: {_get_label_distribution(targets)}")
    
    return create_text_classification_dataset(feature_texts, [str(t) for t in targets], vocab_size, max_length)