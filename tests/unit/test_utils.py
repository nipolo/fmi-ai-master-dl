"""Unit tests for the shared device/seeding helpers."""

import random
import unittest

import numpy as np
import torch

from objdetect.utils import resolve_device, seed_everything


class TestSeedEverything(unittest.TestCase):
    def test_when_seeded_with_same_value_then_rng_draws_match(self):
        # Arrange
        seed = 123

        # Act
        seed_everything(seed)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        seed_everything(seed)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        # Assert
        self.assertEqual(actual, expected)


class TestResolveDevice(unittest.TestCase):
    def test_when_device_given_explicitly_then_it_is_honored(self):
        # Arrange
        requested = "cpu"
        expected_type = "cpu"

        # Act
        actual_device = resolve_device(requested)

        # Assert
        self.assertIsInstance(actual_device, torch.device)
        self.assertEqual(actual_device.type, expected_type)


if __name__ == "__main__":
    unittest.main()
