import unittest
import os
import shutil
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import onnxruntime

try:
    from text_classifier.classifier import TextClassifier
    from text_classifier.strategies_ref import (
        TensorFlowStrategy,
        PyTorchStrategy,
        ScikitLearnStrategy,
    )
except ImportError:
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from text_classifier.classifier import TextClassifier
    from text_classifier.strategies_ref import (
        TensorFlowStrategy,
        PyTorchStrategy,
        ScikitLearnStrategy,
    )


BINARY_TRAINING_DATA_CSV_PATH: str = (
    "../models_multiclass/sms_spam_classifier/training_data.csv"
)
MULTICLASS_TRAINING_DATA_CSV_PATH: str = (
    "../models_multiclass/movie_sentiment_clf/training_data.csv"
)
MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH: str = (
    "../models_multiclass/cabinet_relevance_scorer/training_data.csv"
)

TEST_STRINGS_BINARY: dict = {
    "Win a free iphone today, join using link.": 1,
    "Hi, I'm a bit late.": 0,
}
TEST_STRINGS_MULTICLASS: dict = {
    "I love this movie!": "Positive",
    "This movie was terrible.": "Negative",
    "It was okay, not great.": "Neutral",
}
TEST_STRINGS_MULTICLASS_MAPPING: dict = {
    "A financial analysis report on market trends embeds a minor note on potential volatility in energy futures due to unspecified governmental reviews of extraction policies.": " Chris Wright",
    "A recent report highlights the need for updated land management policies in North Dakota, with indirect references to potential impacts on national park oversight.": "Doug Burgum",
}  # Similar to abov


# --- End Test Configuration ---


class BaseClassifierTest(unittest.TestCase):
    temp_model_dir = "temp_test_models"
    max_features = 100  # Use fewer features for faster tests
    language = "english"

    def setUp(self):
        self.output_path = Path(self.temp_model_dir) / str(uuid.uuid4())
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model_prefix = str(self.output_path / "test_model")
        self.onnx_path = str(self.output_path / "test_model.onnx")

    def tearDown(self):
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)

    def _get_class_labels_and_num_classes(self, csv_path: str):
        if not csv_path or not Path(csv_path).exists():
            self.skipTest(f"Training data CSV not provided or not found: {csv_path}")
        df = pd.read_csv(csv_path)
        class_labels = sorted(list(df["label"].astype(str).unique()))
        num_classes = len(class_labels)
        return class_labels, num_classes

    def _common_test_flow(
        self,
        strategy_class,
        training_data_path: str,
        test_strings_map: dict,
        expected_num_classes_for_proba: int,
    ):
        if not training_data_path:
            self.skipTest(
                f"Training data path not provided for {strategy_class.__name__}"
            )
        if not test_strings_map:
            self.skipTest(f"Test strings not provided for {strategy_class.__name__}")

        class_labels, num_classes_from_data = self._get_class_labels_and_num_classes(
            training_data_path
        )

        self.assertEqual(
            num_classes_from_data,
            expected_num_classes_for_proba,
            f"Data has {num_classes_from_data} unique labels, but test expects {expected_num_classes_for_proba}. Ensure data matches test type.",
        )

        strategy = strategy_class(
            input_dim=self.max_features, num_classes=num_classes_from_data
        )
        classifier = TextClassifier(
            strategy=strategy,
            class_labels=class_labels,  # Use labels from data
            max_features=self.max_features,
            language=self.language,
        )

        # 1. Train model
        print(f"\nTraining {strategy_class.__name__} for {self._testMethodName}...")
        training_metrics = classifier.train(training_data_path)
        self.assertIn("accuracy", training_metrics)
        # Use last epoch accuracy for TF/PyTorch if it's a list
        acc = training_metrics["accuracy"]
        final_accuracy = acc[-1] if isinstance(acc, list) else acc
        self.assertGreaterEqual(
            final_accuracy,
            0.20,
            f"Training accuracy {final_accuracy:.4f} < 0.90 for {strategy_class.__name__}",
        )
        print(
            f"Training complete for {strategy_class.__name__}. Accuracy: {final_accuracy:.4f}"
        )

        # 2. Save model (implicitly tested by ONNX export and explicit load later)
        classifier.save(self.model_prefix)

        # 3. Export to ONNX
        print(f"Exporting {strategy_class.__name__} to ONNX...")
        dummy_onnx_input = np.zeros(
            (1, classifier.fitted_input_features), dtype=np.float32
        )
        classifier.strategy.export_to_onnx(self.onnx_path, X_sample=dummy_onnx_input)
        self.assertTrue(os.path.exists(self.onnx_path))
        print(f"ONNX export complete for {strategy_class.__name__}")

        # 4. Load original model (TextClassifier)
        print(f"Loading original {strategy_class.__name__} TextClassifier...")
        loaded_classifier = TextClassifier.load(self.model_prefix)
        self.assertIsNotNone(loaded_classifier)
        self.assertEqual(loaded_classifier.class_labels, class_labels)
        print(f"Original TextClassifier loaded for {strategy_class.__name__}")

        # 5. Test predictions with loaded original model
        test_texts_orig = list(test_strings_map.keys())
        predictions_orig = loaded_classifier.predict(test_texts_orig)
        probabilities_orig = loaded_classifier.predict_proba(test_texts_orig)

        self.assertEqual(len(predictions_orig), len(test_texts_orig))
        self.assertIsInstance(predictions_orig[0], str)  # predict returns string labels
        self.assertEqual(
            probabilities_orig.shape, (len(test_texts_orig), num_classes_from_data)
        )
        self.assertTrue(
            np.all(probabilities_orig >= 0) and np.all(probabilities_orig <= 1)
        )

        for i, text in enumerate(test_texts_orig):
            self.assertEqual(
                predictions_orig[i],
                test_strings_map[text],
                f"Original model prediction error for '{text}' with {strategy_class.__name__}",
            )
        print(
            f"Predictions with original loaded TextClassifier for {strategy_class.__name__} successful."
        )

        # 6. Load ONNX model and test predictions
        print(f"Loading ONNX model for {strategy_class.__name__}...")
        ort_session = onnxruntime.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )
        input_name = ort_session.get_inputs()[
            0
        ].name  # Should be 'float_input' based on strategies_ref
        output_name = ort_session.get_outputs()[
            0
        ].name  # Should be 'output' based on strategies_ref

        # Preprocess test strings using the loaded_classifier's vectorizer
        # This is crucial: ONNX model expects numerical input
        X_tfidf_test_onnx = loaded_classifier.vectorizer.transform(test_texts_orig)
        if hasattr(X_tfidf_test_onnx, "toarray"):  # If sparse
            X_tfidf_test_onnx = X_tfidf_test_onnx.toarray()
        X_tfidf_test_onnx = X_tfidf_test_onnx.astype(np.float32)

        onnx_predictions_proba = ort_session.run(
            [output_name], {input_name: X_tfidf_test_onnx}
        )[0]
        self.assertEqual(
            onnx_predictions_proba.shape, (len(test_texts_orig), num_classes_from_data)
        )
        self.assertTrue(
            np.all(onnx_predictions_proba >= 0) and np.all(onnx_predictions_proba <= 1)
        )

        # Convert ONNX probabilities to labels using loaded_classifier's label_encoder
        onnx_predicted_int_labels = np.argmax(onnx_predictions_proba, axis=1)
        onnx_predicted_str_labels = loaded_classifier.label_encoder.inverse_transform(
            onnx_predicted_int_labels
        )

        for i, text in enumerate(test_texts_orig):
            self.assertEqual(
                onnx_predicted_str_labels[i],
                test_strings_map[text],
                f"ONNX model prediction error for '{text}' with {strategy_class.__name__}",
            )
        print(f"Predictions with ONNX model for {strategy_class.__name__} successful.")


# --- Test Class for Binary Output (Raw Probability Focus) ---
@unittest.skipIf(
    BINARY_TRAINING_DATA_CSV_PATH is None, "Binary training data path not set."
)
class BinaryRawProbTests(BaseClassifierTest):
    EXPECTED_N_CLASSES_FOR_PROBA = 2  # Binary: 0 or 1 (or two distinct labels)

    def test_scikit_binary_probas(self):
        print(f"\n--- test_scikit_binary_probas ---")
        self._common_test_flow(
            ScikitLearnStrategy,
            BINARY_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_BINARY,
            self.EXPECTED_N_CLASSES_FOR_PROBA,
        )

    def test_tensorflow_binary_probas(self):
        print(f"\n--- test_tensorflow_binary_probas ---")
        self._common_test_flow(
            TensorFlowStrategy,
            BINARY_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_BINARY,
            self.EXPECTED_N_CLASSES_FOR_PROBA,
        )

    def test_pytorch_binary_probas(self):
        print(f"\n--- test_pytorch_binary_probas ---")
        self._common_test_flow(
            PyTorchStrategy,
            BINARY_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_BINARY,
            self.EXPECTED_N_CLASSES_FOR_PROBA,
        )


# --- Test Class for Multiclass (Model Picks One Label) ---
@unittest.skipIf(
    MULTICLASS_TRAINING_DATA_CSV_PATH is None, "Multiclass training data path not set."
)
class MulticlassOneLabelTests(BaseClassifierTest):
    # For this test, we need to determine N_CLASSES from the data
    # The common_test_flow will handle it. User needs to provide data with >2 classes.

    def test_scikit_multiclass_one_label(self):
        print(f"\n--- test_scikit_multiclass_one_label ---")
        _, num_classes = self._get_class_labels_and_num_classes(
            MULTICLASS_TRAINING_DATA_CSV_PATH
        )
        self.assertGreater(
            num_classes, 1, "Multiclass test requires more than 1 class in data."
        )  # Data must have at least 2 for any classification
        self._common_test_flow(
            ScikitLearnStrategy,
            MULTICLASS_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_MULTICLASS,
            num_classes,
        )

    def test_tensorflow_multiclass_one_label(self):
        print(f"\n--- test_tensorflow_multiclass_one_label ---")
        _, num_classes = self._get_class_labels_and_num_classes(
            MULTICLASS_TRAINING_DATA_CSV_PATH
        )
        self.assertGreater(num_classes, 1)
        self._common_test_flow(
            TensorFlowStrategy,
            MULTICLASS_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_MULTICLASS,
            num_classes,
        )

    def test_pytorch_multiclass_one_label(self):
        print(f"\n--- test_pytorch_multiclass_one_label ---")
        _, num_classes = self._get_class_labels_and_num_classes(
            MULTICLASS_TRAINING_DATA_CSV_PATH
        )
        self.assertGreater(num_classes, 1)
        self._common_test_flow(
            PyTorchStrategy,
            MULTICLASS_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_MULTICLASS,
            num_classes,
        )


# --- Test Class for Multiclass (Model Creates Probability Mapping) ---
@unittest.skipIf(
    MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH is None,
    "Multiclass mapping training data path not set.",
)
class MulticlassProbMapTests(BaseClassifierTest):
    # This is essentially the same as MulticlassOneLabelTests in terms of model an_classes = self._get_class_labels_and_num_classes(MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH)
    # output structure of predict_proba. The "mapping" is a client-side interpretation.

    def test_scikit_multiclass_prob_map(self):
        print(f"\n--- test_scikit_multiclass_prob_map ---")
        class_labels, num_classes = self._get_class_labels_and_num_classes(
            MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH
        )
        self.assertGreater(num_classes, 1)
        self._common_test_flow(
            ScikitLearnStrategy,
            MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_MULTICLASS_MAPPING,
            num_classes,
        )

    def test_tensorflow_multiclass_prob_map(self):
        print(f"\n--- test_tensorflow_multiclass_prob_map ---")
        class_labels, num_classes = self._get_class_labels_and_num_classes(
            MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH
        )
        self.assertGreater(num_classes, 1)
        self._common_test_flow(
            TensorFlowStrategy,
            MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_MULTICLASS_MAPPING,
            num_classes,
        )

    def test_pytorch_multiclass_prob_map(self):
        print(f"\n--- test_pytorch_multiclass_prob_map ---")
        class_labels, num_classes = self._get_class_labels_and_num_classes(
            MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH
        )
        self.assertGreater(num_classes, 1)
        self._common_test_flow(
            PyTorchStrategy,
            MULTICLASS_MAPPING_TRAINING_DATA_CSV_PATH,
            TEST_STRINGS_MULTICLASS_MAPPING,
            num_classes,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
