import unittest

import torch
from torch import nn

from projects import rnn


class TestInit(unittest.TestCase):

    def test_when_called_then_inits_parameters_using_uniform_distribution(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        expected_shape_weight_ih = (hidden_size, input_size)
        expected_shape_weight_hh = (hidden_size, hidden_size)
        expected_shape_bias_ih = (hidden_size, )
        expected_shape_bias_hh = (hidden_size, )
        expected_mean = 0
        k = (1 / hidden_size)**0.5
        expected_min = -k
        expected_max = k
        model_torch = nn.RNNCell(input_size, hidden_size)
        expected_num_params = len(list(model_torch.named_parameters()))

        # Act
        model_ours = rnn.RNNCell(input_size, hidden_size)
        actual_shape_weight_ih = model_ours.weight_ih.shape
        actual_shape_weight_hh = model_ours.weight_hh.shape
        actual_shape_bias_ih = model_ours.bias_ih.shape
        actual_shape_bias_hh = model_ours.bias_hh.shape
        actual_mean_weight = round(model_ours.weight_ih.mean().item())
        actual_mean_weight = round(model_ours.weight_hh.mean().item())
        actual_mean_bias = round(model_ours.bias_ih.mean().item())
        actual_mean_bias = round(model_ours.bias_hh.mean().item())
        actual_weight_ih_min = model_ours.weight_ih.min()
        actual_weight_ih_max = model_ours.weight_ih.max()
        actual_weight_hh_min = model_ours.weight_hh.min()
        actual_weight_hh_max = model_ours.weight_hh.max()
        actual_bias_ih_min = model_ours.bias_ih.min()
        actual_bias_ih_max = model_ours.bias_ih.max()
        actual_bias_hh_min = model_ours.bias_hh.min()
        actual_bias_hh_max = model_ours.bias_hh.max()
        actual_num_params = len(list(model_ours.named_parameters()))

        # Assert
        self.assertEqual(actual_num_params, expected_num_params)
        self.assertTupleEqual(actual_shape_weight_ih, expected_shape_weight_ih)
        self.assertTupleEqual(actual_shape_weight_hh, expected_shape_weight_hh)
        self.assertTupleEqual(actual_shape_bias_ih, expected_shape_bias_ih)
        self.assertTupleEqual(actual_shape_bias_hh, expected_shape_bias_hh)
        self.assertGreater(actual_weight_ih_min, expected_min)
        self.assertLess(actual_weight_ih_max, expected_max)
        self.assertGreater(actual_weight_hh_min, expected_min)
        self.assertLess(actual_weight_hh_max, expected_max)
        self.assertGreater(actual_bias_ih_min, expected_min)
        self.assertLess(actual_bias_ih_max, expected_max)
        self.assertGreater(actual_bias_hh_min, expected_min)
        self.assertLess(actual_bias_hh_max, expected_max)
        self.assertEqual(actual_mean_weight, expected_mean)
        self.assertEqual(actual_mean_weight, expected_mean)
        self.assertEqual(actual_mean_bias, expected_mean)
        self.assertEqual(actual_mean_bias, expected_mean)


class TestForward(unittest.TestCase):

    def test_when_memory_provided_then_applies_cell_logic(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        seq_len = 6
        batch_size = 3
        model_torch = nn.RNNCell(input_size, hidden_size)
        xs = torch.randn(seq_len, batch_size, input_size)
        hx = torch.randn(batch_size, hidden_size)
        expected_hidden = model_torch(xs[0], hx)

        # Act
        model_ours = rnn.RNNCell(input_size, hidden_size)
        with torch.no_grad():
            model_ours.weight_ih.copy_(model_torch.weight_ih.detach())
            model_ours.weight_hh.copy_(model_torch.weight_hh.detach())
            model_ours.bias_ih.copy_(model_torch.bias_ih.detach())
            model_ours.bias_hh.copy_(model_torch.bias_hh.detach())
        actual_hidden = model_ours(xs[0], hx)

        # Assert
        torch.testing.assert_close(actual_hidden, expected_hidden)

    def test_when_memory_not_provided_then_applies_cell_logic(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        seq_len = 6
        batch_size = 3
        model_torch = nn.RNNCell(input_size, hidden_size)
        xs = torch.randn(seq_len, batch_size, input_size)
        expected_hidden = model_torch(xs[0])

        # Act
        model_ours = rnn.RNNCell(input_size, hidden_size)
        with torch.no_grad():
            model_ours.weight_ih.copy_(model_torch.weight_ih.detach())
            model_ours.weight_hh.copy_(model_torch.weight_hh.detach())
            model_ours.bias_ih.copy_(model_torch.bias_ih.detach())
            model_ours.bias_hh.copy_(model_torch.bias_hh.detach())
        actual_hidden = model_ours(xs[0])

        # Assert
        torch.testing.assert_close(actual_hidden, expected_hidden)
