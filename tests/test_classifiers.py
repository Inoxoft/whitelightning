import unittest
import os
import shutil
from pathlib import Path
import uuid
import pandas as pd

try:
    from text_classifier.train import TextClassifierRunner
except ImportError:
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from text_classifier.train import TextClassifierRunner


import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BINARY_TRAINING_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "tests/data/binary/training_data.csv")
MULTICLASS_TRAINING_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "tests/data/multiclass/training_data.csv")
MULTILABEL_TRAINING_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "tests/data/multilabel/training_data.csv")

BINARY_TESTING_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "tests/data/binary/testing_data.csv")
MULTICLASS_TESTING_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "tests/data/multiclass/testing_data.csv")
MULTILABEL_TESTING_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "tests/data/multilabel/testing_data.csv")


class TextClassifierRunnerTest(unittest.TestCase):
    temp_model_dir = "temp_test_models"

    def setUp(self):
        self.output_path = Path(self.temp_model_dir) / str(uuid.uuid4())
        self.output_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)

    def _test_model(
        self, data_type: str, library_type: str, train_path: str, test_path: str
    ):
        # Read unique labels from training data
        train_df = pd.read_csv(train_path)
        labels = sorted(train_df["label"].unique().tolist())

        # Initialize runner
        runner = TextClassifierRunner(
            data_type=data_type,
            library_type=library_type,
            train_path=train_path,
            test_path=test_path,
            labels=labels,
            output_path=str(self.output_path),
        )

        # Train model
        runner.train_model()

        # Test predictions
        test_df = pd.read_csv(test_path)
        test_texts = test_df["text"].values.tolist()

        predictions = runner.predict(test_texts)

        if data_type == "multilabel":
            print(predictions)
            # For multilabel, use threshold of 0.5
            predictions = [[1 if p >= 0.5 else 0 for p in pred] for pred in predictions]
        else:
            # For binary and multiclass, take argmax
            predictions = predictions.argmax(axis=1) if predictions.ndim > 1 else (predictions > 0.5).astype(int)

        self.assertEqual(len(predictions), len(test_texts))

        if data_type == "multilabel":
            for pred in predictions:
                label_indices = [i for i, p in enumerate(pred) if p == 1]
                pred_labels = [labels[i] for i in label_indices]
                self.assertTrue(any(label in labels for label in pred_labels))
        else:
            for pred in predictions:
                self.assertIn(labels[pred], labels)

    def test_binary_classification(self):
        """Test binary classification with all libraries"""
        for library in ["torch", "tensorflow", "sklearn"]:
            with self.subTest(library=library):
                self._test_model(
                    data_type="binary",
                    library_type=library,
                    train_path=BINARY_TRAINING_DATA_CSV_PATH,
                    test_path=BINARY_TESTING_DATA_CSV_PATH,
                )

    def test_multiclass_classification(self):
        """Test multiclass classification with all libraries"""
        for library in ["torch", "tensorflow", "sklearn"]:
            with self.subTest(library=library):
                self._test_model(
                    data_type="multiclass",
                    library_type=library,
                    train_path=MULTICLASS_TRAINING_DATA_CSV_PATH,
                    test_path=MULTICLASS_TESTING_DATA_CSV_PATH,
                )

    def test_multilabel_classification(self):
        """Test multilabel classification with all libraries"""
        for library in ["torch", "tensorflow", "sklearn"]:
            with self.subTest(library=library):
                self._test_model(
                    data_type="multilabel",
                    library_type=library,
                    train_path=MULTILABEL_TRAINING_DATA_CSV_PATH,
                    test_path=MULTILABEL_TESTING_DATA_CSV_PATH,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
