import unittest

import torch

from src.model import CNN


class TestCNN(unittest.TestCase):
    def test_forward_pass_returns_ten_class_logits(self):
        model = CNN(num_classes=10)
        images = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            predictions = model(images)

        self.assertEqual(tuple(predictions.shape), (2, 10))
        self.assertTrue(torch.isfinite(predictions).all().item())


if __name__ == "__main__":
    unittest.main()
