import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from omicpro.artifacts import ExperimentLayout, restore_preprocessor, serialize_preprocessor
from omicpro.check_data import check_config
from omicpro.data_utils import _extract_fold_map, _prepare_modality_df
from omicpro.model_registry import available_models
from omicpro.preprocess import FoldPreprocessor
from omicpro.train_cv import load_config


class StructureTests(unittest.TestCase):
    def test_experiment_layout(self):
        layout = ExperimentLayout.from_config(
            {"run_name": "rice_demo", "model_dir": "models", "result_dir": "results"}
        )
        self.assertEqual(str(layout.checkpoint_path("yd", 2)), str(layout.model_root / "k2" / "yd" / "omicpro.pt"))
        self.assertEqual(str(layout.fold_prediction_path("yd", 2)), str(layout.result_root / "k2" / "yd.csv"))

    def test_model_registry(self):
        self.assertIn("omicpro", available_models())

    def test_duplicate_sample_ids_are_rejected(self):
        frame = pd.DataFrame({"ID": ["A", "A"], "x": [1, 2]})
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            _prepare_modality_df(frame, "ID")

    def test_repeated_cv_columns_require_an_explicit_choice(self):
        frame = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "cv_1": [1, 2, 3],
                "cv_2": [3, 1, 2],
            }
        )
        with self.assertRaisesRegex(ValueError, "fold_column"):
            _extract_fold_map(frame, "ID")

        selected = _extract_fold_map(frame, "ID", fold_column="cv_2")
        self.assertEqual(selected.to_dict(), {"A": 3, "B": 1, "C": 2})

    def test_one_hot_cv_columns_are_still_supported(self):
        frame = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "fold_1": [1, 0, 0],
                "fold_2": [0, 1, 0],
                "fold_3": [0, 0, 1],
            }
        )
        selected = _extract_fold_map(frame, "ID")
        self.assertEqual(selected.to_dict(), {"A": 1, "B": 2, "C": 3})

    def test_preprocessor_round_trip(self):
        index = ["A", "B", "C"]
        genotype = pd.DataFrame([[0, 1], [1, np.nan], [2, 0]], index=index, columns=["g1", "g2"])
        expression = pd.DataFrame([[1, 2], [2, 4], [3, 6]], index=index, columns=["e1", "e2"])
        metabolites = pd.DataFrame([[3], [4], [5]], index=index, columns=["m1"])
        original = FoldPreprocessor(topk_expression=1).fit(genotype, expression, metabolites)
        restored = restore_preprocessor(serialize_preprocessor(original))

        pairs = (
            (original.transform_genotype(genotype), restored.transform_genotype(genotype)),
            (original.transform_expression(expression), restored.transform_expression(expression)),
            (original.transform_metabolites(metabolites), restored.transform_metabolites(metabolites)),
        )
        for expected, actual in pairs:
            np.testing.assert_allclose(expected.to_numpy(), actual.to_numpy())

    def test_config_data_path_is_resolved_relative_to_config(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "experiment.yaml"
            config_path.write_text("data_dir: data/private\ntraits: [target]\n", encoding="utf-8")
            cfg = load_config(config_path)
            self.assertEqual(Path(cfg["data_dir"]), Path(tmp) / "data" / "private")

    def test_public_demo_data_matches_its_config(self):
        package_root = Path(__file__).resolve().parents[1]
        report = check_config(package_root / "configs" / "demo.yaml")
        self.assertEqual(report["samples"], 12)
        self.assertEqual(report["traits"], ["demo_trait"])
        self.assertEqual(report["fold_counts"], {"1": 4, "2": 4, "3": 4})


if __name__ == "__main__":
    unittest.main()
