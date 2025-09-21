"""
Tests for Part-of-Speech Tagging functionality.

This module tests:
1. POS dataset loading and processing
2. POS tagger model forward pass and training
3. Training pipeline and heuristics
4. Metrics computation
"""

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from adaptiveneuralnetwork.data.kaggle_datasets import (
    POSDataset,
    _build_pos_vocabularies,
    _compute_pos_statistics,
    _load_pos_data,
    load_pos_tagging_dataset,
)
from adaptiveneuralnetwork.models.pos_tagger import (
    POSTagger,
    POSTaggerConfig,
    compute_sequence_lengths,
    compute_token_accuracy,
    masked_cross_entropy_loss,
)


class TestPOSDataLoading(unittest.TestCase):
    """Test POS dataset loading functionality."""

    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "test_pos.csv"

        # Create test CSV data
        test_data = [
            {"sentence": 1, "word": "The", "pos": "DT"},
            {"sentence": 1, "word": "cat", "pos": "NN"},
            {"sentence": 1, "word": "sat", "pos": "VBD"},
            {"sentence": 2, "word": "Dogs", "pos": "NNS"},
            {"sentence": 2, "word": "bark", "pos": "VBP"},
            {"sentence": 3, "word": "I", "pos": "PRP"},
            {"sentence": 3, "word": "love", "pos": "VBP"},
            {"sentence": 3, "word": "coding", "pos": "VBG"},
        ]

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sentence", "word", "pos"])
            writer.writeheader()
            writer.writerows(test_data)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_pos_data(self):
        """Test basic POS data loading from CSV."""
        sentences, tags = _load_pos_data(str(self.csv_path))

        expected_sentences = [["The", "cat", "sat"], ["Dogs", "bark"], ["I", "love", "coding"]]
        expected_tags = [["DT", "NN", "VBD"], ["NNS", "VBP"], ["PRP", "VBP", "VBG"]]

        self.assertEqual(sentences, expected_sentences)
        self.assertEqual(tags, expected_tags)

    def test_compute_pos_statistics(self):
        """Test statistics computation."""
        sentences = [["The", "cat"], ["Dogs", "bark", "loudly"]]
        tags = [["DT", "NN"], ["NNS", "VBP", "RB"]]

        stats = _compute_pos_statistics(sentences, tags)

        self.assertEqual(stats["num_sentences"], 2)
        self.assertEqual(stats["total_tokens"], 5)
        self.assertEqual(stats["unique_tokens"], 5)
        self.assertEqual(stats["unique_tags"], 5)  # DT, NN, NNS, VBP, RB
        self.assertEqual(stats["avg_sentence_length"], 2.5)
        self.assertEqual(stats["max_sentence_length"], 3)

    def test_build_pos_vocabularies(self):
        """Test vocabulary building."""
        sentences = [["The", "cat", "sat"], ["The", "dog", "ran"]]
        tags = [["DT", "NN", "VBD"], ["DT", "NN", "VBD"]]

        vocab, tag_vocab = _build_pos_vocabularies(sentences, tags, vocab_size=10)

        # Check special tokens
        self.assertEqual(vocab["<PAD>"], 0)
        self.assertEqual(vocab["<UNK>"], 1)
        self.assertEqual(tag_vocab["<PAD>"], 0)

        # Check that frequent tokens are included
        self.assertIn("The", vocab)
        self.assertIn("cat", vocab)

        # Check all tags are included
        self.assertIn("DT", tag_vocab)
        self.assertIn("NN", tag_vocab)
        self.assertIn("VBD", tag_vocab)

    def test_load_pos_tagging_dataset(self):
        """Test complete dataset loading."""
        result = load_pos_tagging_dataset(
            str(self.csv_path), max_sentences=2, vocab_size=50, seed=42
        )

        self.assertIn("datasets", result)
        self.assertIn("vocab", result)
        self.assertIn("tag_vocab", result)
        self.assertIn("stats", result)

        # Check dataset splits
        datasets = result["datasets"]
        self.assertIn("train", datasets)
        self.assertIn("val", datasets)
        self.assertIn("test", datasets)

        # Check dataset functionality
        train_dataset = datasets["train"]
        self.assertIsInstance(train_dataset, POSDataset)
        self.assertGreater(len(train_dataset), 0)

        # Test getting an item
        item = train_dataset[0]
        self.assertIn("tokens", item)
        self.assertIn("tags", item)
        self.assertIn("token_ids", item)
        self.assertIn("tag_ids", item)


class TestPOSDataset(unittest.TestCase):
    """Test POSDataset class functionality."""

    def setUp(self):
        """Set up test data."""
        self.sentences = [["The", "cat", "sat"], ["Dogs", "bark"]]
        self.tags = [["DT", "NN", "VBD"], ["NNS", "VBP"]]
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "The": 2, "cat": 3, "sat": 4, "Dogs": 5, "bark": 6}
        self.tag_vocab = {"<PAD>": 0, "DT": 1, "NN": 2, "VBD": 3, "NNS": 4, "VBP": 5}

        self.dataset = POSDataset(self.sentences, self.tags, self.vocab, self.tag_vocab)

    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), 2)

    def test_get_item(self):
        """Test getting individual items."""
        item = self.dataset[0]

        self.assertEqual(item["tokens"], ["The", "cat", "sat"])
        self.assertEqual(item["tags"], ["DT", "NN", "VBD"])
        self.assertEqual(item["token_ids"], [2, 3, 4])
        self.assertEqual(item["tag_ids"], [1, 2, 3])
        self.assertEqual(item["length"], 3)

    def test_get_batch(self):
        """Test batch creation with padding."""
        batch = self.dataset.get_batch([0, 1])

        self.assertEqual(len(batch["input_ids"]), 2)
        self.assertEqual(len(batch["tag_ids"]), 2)
        self.assertEqual(len(batch["lengths"]), 2)

        # Check padding (second sequence should be padded to length 3)
        self.assertEqual(len(batch["input_ids"][0]), 3)  # First sequence length
        self.assertEqual(len(batch["input_ids"][1]), 3)  # Padded to same length
        self.assertEqual(batch["input_ids"][1][2], 0)  # Padding token

        # Check attention mask
        self.assertEqual(batch["attention_mask"][0], [1, 1, 1])  # All real tokens
        self.assertEqual(batch["attention_mask"][1], [1, 1, 0])  # Last token is padding


class TestPOSTaggerModel(unittest.TestCase):
    """Test POS tagger model functionality."""

    def setUp(self):
        """Set up test model."""
        self.config = POSTaggerConfig(
            vocab_size=100,
            num_tags=20,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=1,
            dropout=0.1,
            model_type="bilstm",
        )
        self.model = POSTagger(self.config)

    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.model, POSTagger)
        self.assertEqual(self.model.config.vocab_size, 100)
        self.assertEqual(self.model.config.num_tags, 20)

    def test_forward_pass_bilstm(self):
        """Test BiLSTM forward pass."""
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        lengths = torch.tensor([5, 3])

        outputs = self.model(input_ids, attention_mask, lengths)

        self.assertIn("logits", outputs)
        self.assertIn("hidden_states", outputs)

        logits = outputs["logits"]
        self.assertEqual(logits.shape, (batch_size, seq_len, 20))

    def test_forward_pass_transformer(self):
        """Test Transformer forward pass."""
        config = POSTaggerConfig(
            vocab_size=100,
            num_tags=20,
            embedding_dim=64,
            hidden_dim=128,
            model_type="transformer",
            num_heads=4,
        )
        model = POSTagger(config)

        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        outputs = model(input_ids)

        self.assertIn("logits", outputs)
        logits = outputs["logits"]
        self.assertEqual(logits.shape, (batch_size, seq_len, 20))

    def test_predict(self):
        """Test prediction method."""
        batch_size, seq_len = 2, 5
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        predictions = self.model.predict(input_ids)

        self.assertEqual(predictions.shape, (batch_size, seq_len))
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < 20))


class TestPOSUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_compute_sequence_lengths(self):
        """Test sequence length computation."""

        input_ids = torch.tensor(
            [
                [1, 2, 3, 0, 0],  # Length 3
                [1, 2, 0, 0, 0],  # Length 2
                [1, 2, 3, 4, 5],  # Length 5
            ]
        )

        lengths = compute_sequence_lengths(input_ids, pad_token_id=0)
        expected = torch.tensor([3, 2, 5])

        self.assertTrue(torch.equal(lengths, expected))

    def test_compute_token_accuracy(self):
        """Test token accuracy computation."""
        predictions = torch.tensor([[1, 2, 3, 0], [1, 1, 0, 0]])
        labels = torch.tensor(
            [
                [1, 2, 2, 0],  # 2/3 correct (ignoring padding)
                [1, 2, 0, 0],  # 1/2 correct (ignoring padding)
            ]
        )
        mask = torch.tensor(
            [
                [1, 1, 1, 0],  # 3 real tokens
                [1, 1, 0, 0],  # 2 real tokens
            ]
        )

        accuracy = compute_token_accuracy(predictions, labels, mask)
        expected = 3.0 / 5.0  # 3 correct out of 5 total real tokens

        self.assertAlmostEqual(accuracy, expected, places=5)

    def test_masked_cross_entropy_loss(self):
        """Test masked cross entropy loss."""
        batch_size, seq_len, num_tags = 2, 3, 4
        logits = torch.randn(batch_size, seq_len, num_tags)
        labels = torch.randint(0, num_tags, (batch_size, seq_len))
        mask = torch.tensor(
            [
                [1, 1, 0],  # 2 real tokens
                [1, 0, 0],  # 1 real token
            ]
        )

        loss = masked_cross_entropy_loss(logits, labels, mask)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Should be scalar
        self.assertGreater(loss.item(), 0)  # Should be positive


class TestTrainingHeuristics(unittest.TestCase):
    """Test training heuristics computation."""

    def test_dynamic_heuristics(self):
        """Test dynamic heuristics computation."""
        # Import the function from the training script
        import sys

        sys.path.append("/home/runner/work/adaptiveneuralnetwork/adaptiveneuralnetwork")

        # Mock args
        class MockArgs:
            epochs = None
            batch_size = None
            gradient_accumulation_steps = 1

        args = MockArgs()

        # Test small dataset
        stats_small = {"num_sentences": 3000, "total_tokens": 50000, "avg_sentence_length": 16.7}

        try:
            from train_pos_tagging import get_dynamic_heuristics

            heuristics = get_dynamic_heuristics(stats_small, args)

            self.assertEqual(heuristics["epochs"], 40)  # Small dataset -> 40 epochs
            self.assertEqual(heuristics["batch_size"], 32)  # Normal batch size

            # Test large dataset
            stats_large = {
                "num_sentences": 50000,
                "total_tokens": 1000000,
                "avg_sentence_length": 20,
            }

            heuristics = get_dynamic_heuristics(stats_large, args)

            self.assertEqual(heuristics["epochs"], 12)  # Large dataset -> 12 epochs
            self.assertEqual(heuristics["batch_size"], 16)  # Reduced batch size for memory

        except ImportError:
            # Skip if we can't import the training script
            self.skipTest("Cannot import training script")


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration."""

    def test_training_smoke_test(self):
        """Smoke test for training pipeline."""
        # Create minimal synthetic data
        sentences = [["The", "cat"], ["Dogs", "bark"]] * 5
        tags = [["DT", "NN"], ["NNS", "VBP"]] * 5

        vocab = {"<PAD>": 0, "<UNK>": 1, "The": 2, "cat": 3, "Dogs": 4, "bark": 5}
        tag_vocab = {"<PAD>": 0, "DT": 1, "NN": 2, "NNS": 3, "VBP": 4}

        # Create dataset splits
        train_sentences = sentences[:8]
        train_tags = tags[:8]
        val_sentences = sentences[8:]
        val_tags = tags[8:]

        train_dataset = POSDataset(train_sentences, train_tags, vocab, tag_vocab)
        val_dataset = POSDataset(val_sentences, val_tags, vocab, tag_vocab)

        # Create model
        config = POSTaggerConfig(
            vocab_size=len(vocab),
            num_tags=len(tag_vocab),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1,
            dropout=0.1,
        )
        model = POSTagger(config)

        # Test forward pass
        batch = train_dataset.get_batch([0, 1])
        input_ids = torch.tensor(batch["input_ids"])
        tag_ids = torch.tensor(batch["tag_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])

        outputs = model(input_ids, attention_mask)
        logits = outputs["logits"]

        # Test loss computation
        loss = masked_cross_entropy_loss(logits, tag_ids, attention_mask)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

        # Test gradient computation
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
