import logging
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

logger = logging.getLogger(__name__)


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
        
        train_df = pd.read_csv(train_path, on_bad_lines="skip")
        
        if data_type == "multilabel":
            
            all_labels = set()
            for label_string in train_df["label"].values:
                individual_labels = [label.strip() for label in str(label_string).split(",")]
                all_labels.update(individual_labels)
            labels = sorted(list(all_labels))
        else:
           
            labels = sorted(train_df["label"].unique().tolist())

        
        runner = TextClassifierRunner(
            data_type=data_type,
            library_type=library_type,
            train_path=train_path,
            test_path=test_path,
            labels=labels,
            output_path=str(self.output_path),
        )

       
        runner.train_model()

       
        test_df = pd.read_csv(test_path)
        test_texts = test_df["text"].values.tolist()

        predictions = runner.predict(test_texts)

        self.assertEqual(len(predictions), len(test_texts))

        if data_type == "multilabel":
            for pred in predictions:
                self.assertTrue(len(pred) <= len(labels))

        logger.info(f"Predictions: {predictions}")
        logger.info(f"Labels: {labels}")

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