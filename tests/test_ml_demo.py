import unittest

from ml_demo import DATASETS, MODELS, train_and_evaluate


class TrainAndEvaluateTests(unittest.TestCase):
    def test_all_datasets_train_and_return_metrics(self):
        for dataset_name in DATASETS:
            with self.subTest(dataset=dataset_name):
                result = train_and_evaluate(dataset_name=dataset_name)
                self.assertGreaterEqual(result["accuracy"], 0)
                self.assertLessEqual(result["accuracy"], 1)
                self.assertEqual(len(result["actual"]), len(result["predictions"]))

    def test_all_models_train_on_iris(self):
        for model_name in MODELS:
            with self.subTest(model=model_name):
                result = train_and_evaluate(model_name=model_name)
                self.assertEqual(len(result["actual"]), len(result["predictions"]))

    def test_rejects_unknown_dataset_or_model(self):
        with self.assertRaises(ValueError):
            train_and_evaluate(dataset_name="Unknown")
        with self.assertRaises(ValueError):
            train_and_evaluate(model_name="Unknown")


if __name__ == "__main__":
    unittest.main()
