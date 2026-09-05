import unittest


class ImportTests(unittest.TestCase):
    def test_model_import(self):
        from omicmap.model import MultiOmicsMultiTaskRegressor

        self.assertIsNotNone(MultiOmicsMultiTaskRegressor)


if __name__ == "__main__":
    unittest.main()
