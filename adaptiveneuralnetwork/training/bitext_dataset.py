"""
Bitext dataset integration with graceful Kaggle fallback and synthetic dataset.

This module provides:
1. Bitext dataset loader using kagglehub when credentials are available
2. Deterministic synthetic in-memory dataset fallback
3. Intent classification dataset structure
"""

import os
import random
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BitextSample:
    """A single sample from the Bitext dataset."""
    text: str
    intent: str
    intent_id: int


class BitextDataset:
    """Bitext dataset for intent classification."""
    
    def __init__(self, samples: List[BitextSample]):
        self.samples = samples
        self.intent_to_id = self._build_intent_mapping()
        self.id_to_intent = {v: k for k, v in self.intent_to_id.items()}
    
    def _build_intent_mapping(self) -> Dict[str, int]:
        """Build mapping from intent names to IDs."""
        unique_intents = sorted(set(sample.intent for sample in self.samples))
        return {intent: idx for idx, intent in enumerate(unique_intents)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> BitextSample:
        return self.samples[idx]
    
    def get_texts_and_labels(self) -> Tuple[List[str], List[int]]:
        """Get texts and labels for sklearn training."""
        texts = [sample.text for sample in self.samples]
        labels = [sample.intent_id for sample in self.samples]
        return texts, labels
    
    @property
    def num_classes(self) -> int:
        return len(self.intent_to_id)
    
    @property
    def class_names(self) -> List[str]:
        return list(self.intent_to_id.keys())


def has_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are available."""
    return (
        os.getenv("KAGGLE_USERNAME") is not None and 
        os.getenv("KAGGLE_KEY") is not None
    )


def load_bitext_dataset_from_kaggle() -> Optional[BitextDataset]:
    """
    Load Bitext dataset from Kaggle using kagglehub.
    
    Returns:
        BitextDataset if successful, None if failed
    """
    if not has_kaggle_credentials():
        logger.info("Kaggle credentials not found (KAGGLE_USERNAME and KAGGLE_KEY)")
        return None
    
    try:
        # Import kagglehub only when needed
        import kagglehub
        import pandas as pd
        
        logger.info("Downloading Bitext dataset from Kaggle...")
        
        # Download the Bitext Customer Service Single Intent dataset
        # This is a common intent classification dataset
        path = kagglehub.dataset_download("bitext/customer-support-intent-dataset")
        
        # Look for CSV files in the downloaded path
        import glob
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if not csv_files:
            logger.error("No CSV files found in downloaded dataset")
            return None
        
        # Load the first CSV file found
        df = pd.read_csv(csv_files[0])
        
        # Expect columns like 'text' and 'intent' or similar
        # Try common column name variations
        text_col = None
        intent_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'text' in col_lower or 'utterance' in col_lower or 'query' in col_lower:
                text_col = col
            elif 'intent' in col_lower or 'category' in col_lower or 'label' in col_lower:
                intent_col = col
        
        if text_col is None or intent_col is None:
            logger.error(f"Could not find text and intent columns. Available columns: {df.columns.tolist()}")
            return None
        
        logger.info(f"Using columns: text='{text_col}', intent='{intent_col}'")
        
        # Create BitextSample objects
        samples = []
        intent_to_id = {}
        next_id = 0
        
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            intent = str(row[intent_col]).strip()
            
            if not text or text.lower() in ['nan', 'none', '']:
                continue
                
            if intent not in intent_to_id:
                intent_to_id[intent] = next_id
                next_id += 1
            
            samples.append(BitextSample(
                text=text,
                intent=intent,
                intent_id=intent_to_id[intent]
            ))
        
        logger.info(f"Loaded {len(samples)} samples with {len(intent_to_id)} unique intents from Kaggle")
        return BitextDataset(samples)
        
    except Exception as e:
        logger.warning(f"Failed to load Bitext dataset from Kaggle: {e}")
        return None


def create_synthetic_bitext_dataset(num_samples: int = 1000, seed: int = 42) -> BitextDataset:
    """
    Create a deterministic synthetic in-memory Bitext dataset.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        BitextDataset with synthetic data
    """
    # Set deterministic seed
    random.seed(seed)
    
    # Define intents and template utterances
    intent_templates = {
        "greeting": [
            "Hello, how are you?",
            "Hi there!",
            "Good morning",
            "Hey, what's up?",
            "Greetings",
            "Good afternoon",
            "Good evening", 
            "How do you do?",
            "Nice to meet you",
            "Welcome"
        ],
        "book_hotel": [
            "I want to book a hotel room",
            "Can you help me find a hotel?",
            "I need accommodation for tonight",
            "Book me a room please",
            "Looking for hotel reservations",
            "I need a place to stay",
            "Find me a hotel near downtown",
            "Reserve a room for two people",
            "I want to make a hotel reservation",
            "Book a suite for the weekend"
        ],
        "cancel_booking": [
            "I want to cancel my booking",
            "Please cancel my reservation",
            "Can I cancel my order?",
            "I need to cancel this",
            "Cancel my booking please",
            "I want to cancel my hotel reservation",
            "Can you cancel my flight?",
            "I need to cancel my appointment",
            "Please cancel my subscription",
            "I want to cancel this transaction"
        ],
        "check_weather": [
            "What's the weather like?",
            "How's the weather today?",
            "Is it going to rain?",
            "Check weather forecast",
            "Tell me about today's weather",
            "Will it be sunny tomorrow?",
            "What's the temperature outside?",
            "Is it cloudy today?",
            "Do I need an umbrella?",
            "What's the weather forecast for this week?"
        ],
        "order_food": [
            "I want to order food",
            "Can I get a pizza?",
            "I'm hungry, what can I order?",
            "Place a food order",
            "I'd like to order something to eat",
            "Can you recommend a restaurant?",
            "I want to order Chinese food",
            "Get me some burgers",
            "I need food delivery",
            "Order me some sushi"
        ],
        "technical_support": [
            "I need help with my account",
            "Something is not working",
            "Can you fix this issue?",
            "I have a technical problem",
            "Need support please",
            "My device is not responding",
            "The app keeps crashing",
            "I can't log into my account",
            "There's a bug in the system",
            "I need customer service"
        ]
    }
    
    # Additional words for variation
    variations = [
        "please", "thanks", "thank you", "help", "urgent", "now", "today", 
        "quickly", "ASAP", "immediately", "soon", "later", "tomorrow"
    ]
    
    samples = []
    intents = list(intent_templates.keys())
    
    for i in range(num_samples):
        # Choose random intent
        intent = intents[i % len(intents)]
        
        # Choose random template
        templates = intent_templates[intent]
        template = templates[i % len(templates)]
        
        # Add some variation
        text = template
        if random.random() < 0.3:  # 30% chance to add variation
            variation = random.choice(variations)
            if random.random() < 0.5:
                text = f"{text}, {variation}"
            else:
                text = f"{variation}, {text}"
        
        samples.append(BitextSample(
            text=text,
            intent=intent,
            intent_id=intents.index(intent)
        ))
    
    logger.info(f"Created synthetic dataset with {num_samples} samples and {len(intents)} intents")
    return BitextDataset(samples)


def load_bitext_dataset(use_synthetic: bool = False, num_samples: int = 1000) -> BitextDataset:
    """
    Load Bitext dataset with graceful fallback.
    
    Args:
        use_synthetic: If True, use synthetic data regardless of Kaggle credentials
        num_samples: Number of samples for synthetic dataset
        
    Returns:
        BitextDataset (from Kaggle or synthetic)
    """
    if use_synthetic:
        logger.info("Using synthetic dataset as requested")
        return create_synthetic_bitext_dataset(num_samples)
    
    # Try to load from Kaggle first
    dataset = load_bitext_dataset_from_kaggle()
    
    if dataset is not None:
        logger.info("Successfully loaded Bitext dataset from Kaggle")
        return dataset
    
    # Fallback to synthetic dataset
    logger.info("Falling back to synthetic dataset")
    return create_synthetic_bitext_dataset(num_samples)