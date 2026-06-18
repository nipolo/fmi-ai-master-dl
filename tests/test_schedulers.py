"""Unit tests for the LR schedulers"""

import unittest

from objdetect.training.schedulers import build_scheduler, sample_lr_curve


class TestSampleLrCurveStep(unittest.TestCase):
    def test_when_step_decay_then_drops_by_gamma_every_step_size(self):
        # Arrange
        epochs = 9
        base_lr = 0.01
        step_size = 3
        gamma = 0.1
        expected_first = 0.01
        expected_after_one_step = 0.001
        expected_length = epochs

        # Act
        lrs = sample_lr_curve(
            "step", epochs, base_lr=base_lr, step_size=step_size, gamma=gamma
        )
        actual_length = len(lrs)

        # Assert
        self.assertEqual(actual_length, expected_length)
        self.assertAlmostEqual(lrs[0], expected_first)
        self.assertAlmostEqual(lrs[3], expected_after_one_step)
        # Staircase: constant within each block of step_size epochs.
        self.assertAlmostEqual(lrs[0], lrs[1])
        self.assertAlmostEqual(lrs[1], lrs[2])


class TestSampleLrCurveCosine(unittest.TestCase):
    def test_when_cosine_annealing_then_decreases_from_base_to_near_min(self):
        # Arrange
        epochs = 12
        base_lr = 0.01
        eta_min = 1e-5
        expected_first = 0.01

        # Act
        lrs = sample_lr_curve("cosine", epochs, base_lr=base_lr, eta_min=eta_min)
        actual_first = lrs[0]
        is_monotonic_non_increasing = all(
            lrs[i] >= lrs[i + 1] - 1e-9 for i in range(len(lrs) - 1)
        )

        # Assert
        self.assertAlmostEqual(actual_first, expected_first)
        self.assertTrue(is_monotonic_non_increasing)
        self.assertLess(lrs[-1], base_lr)
        self.assertGreaterEqual(lrs[-1], eta_min)


class TestBuildScheduler(unittest.TestCase):
    def test_when_unknown_name_then_raises_value_error(self):
        # Arrange
        import torch

        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)

        # Act / Assert
        with self.assertRaises(ValueError):
            build_scheduler("triangular", optimizer, epochs=5)


if __name__ == "__main__":
    unittest.main()
