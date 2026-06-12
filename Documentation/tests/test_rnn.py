import unittest

import torch
from torch import nn

from projects import rnn


class TestInit(unittest.TestCase):

    def test_when_num_layers_not_specified_then_creates_one_layer(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        expected_num_layers = 1
        expected_layers_cls = nn.ModuleList

        # Act
        model = rnn.RNN(input_size, hidden_size)
        actual_layers = model.cell_layers
        actual_num_layers = len(actual_layers)

        # Assert
        self.assertEqual(actual_num_layers, expected_num_layers)
        self.assertIsInstance(actual_layers, expected_layers_cls)

    def test_when_num_layers_specified_then_creates_num_layers_cell_layers(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        expected_num_layers = 5

        # Act
        model = rnn.RNN(input_size, hidden_size, expected_num_layers)
        actual_num_layers = len(model.cell_layers)

        # Assert
        self.assertEqual(actual_num_layers, expected_num_layers)

    def test_when_batch_first_passed_then_saves_to_state(self):
        # Arrange
        expected_num_layers = 10
        input_size = 6
        hidden_size = 10
        expected_batch_first = True

        # Act
        model = rnn.RNN(input_size, hidden_size, expected_num_layers, expected_batch_first)
        actual_batch_first = model.batch_first

        # Assert
        self.assertEqual(actual_batch_first, expected_batch_first)


class TestForward(unittest.TestCase):

    def test_when_memory_provided_then_applies_rnn_logic(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        num_layers = 2
        xs = torch.randn(5, 3, 10)
        h0 = torch.randn(2, 3, 20)
        model_torch = nn.RNN(input_size, hidden_size, num_layers)
        expected_output, expected_hn = model_torch(xs, h0)

        model_ours = rnn.RNN(input_size, hidden_size, num_layers)
        model_torch_params = dict(model_torch.named_parameters())

        with torch.no_grad():
            model_ours.cell_layers[0].weight_ih.copy_(model_torch_params['weight_ih_l0'].detach())
            model_ours.cell_layers[0].weight_hh.copy_(model_torch_params['weight_hh_l0'].detach())
            model_ours.cell_layers[0].bias_ih.copy_(model_torch_params['bias_ih_l0'].detach())
            model_ours.cell_layers[0].bias_hh.copy_(model_torch_params['bias_hh_l0'].detach())
            model_ours.cell_layers[1].weight_ih.copy_(model_torch_params['weight_ih_l1'].detach())
            model_ours.cell_layers[1].weight_hh.copy_(model_torch_params['weight_hh_l1'].detach())
            model_ours.cell_layers[1].bias_ih.copy_(model_torch_params['bias_ih_l1'].detach())
            model_ours.cell_layers[1].bias_hh.copy_(model_torch_params['bias_hh_l1'].detach())

        # Act
        actual_output, actual_hn = model_ours(xs, h0)

        # Assert
        torch.testing.assert_close(actual_output, expected_output)
        torch.testing.assert_close(actual_hn, expected_hn)

    def test_when_backward_called_then_computes_gradients(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        num_layers = 2
        batch_size = 3
        seq_len = 5

        model = rnn.RNN(input_size, hidden_size, num_layers)
        xs = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(num_layers, batch_size, hidden_size, requires_grad=True)
        out, hn = model(xs, h0)
        loss = out.sum() + hn.sum()

        # Act
        loss.backward()

        # Assert
        self.assertIsNotNone(h0.grad)
        self.assertIsNotNone(model.cell_layers[0].weight_ih.grad)
        self.assertIsNotNone(model.cell_layers[0].weight_hh.grad)

    def test_when_batch_first_true_then_applies_rnn_logic(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        num_layers = 2
        batch_size = 3
        seq_len = 5
        batch_first = True
        rnn_torch = nn.RNN(input_size, hidden_size, num_layers, batch_first=batch_first)
        torch_parameters = dict(rnn_torch.named_parameters())
        xs_torch = torch.randn(batch_size, seq_len, input_size)
        h0_torch = torch.randn(num_layers, batch_size, hidden_size)
        expected_out, expected_hn = rnn_torch(xs_torch, h0_torch)

        rnn_ours = rnn.RNN(input_size, hidden_size, num_layers, batch_first=batch_first)
        with torch.no_grad():
            rnn_ours.cell_layers[0].weight_hh.copy_(torch_parameters['weight_hh_l0'].detach())
            rnn_ours.cell_layers[0].weight_ih.copy_(torch_parameters['weight_ih_l0'].detach())
            rnn_ours.cell_layers[0].bias_ih.copy_(torch_parameters['bias_ih_l0'].detach())
            rnn_ours.cell_layers[0].bias_hh.copy_(torch_parameters['bias_hh_l0'].detach())
            rnn_ours.cell_layers[1].weight_hh.copy_(torch_parameters['weight_hh_l1'].detach())
            rnn_ours.cell_layers[1].weight_ih.copy_(torch_parameters['weight_ih_l1'].detach())
            rnn_ours.cell_layers[1].bias_ih.copy_(torch_parameters['bias_ih_l1'].detach())
            rnn_ours.cell_layers[1].bias_hh.copy_(torch_parameters['bias_hh_l1'].detach())
        xs_ours = xs_torch.clone()
        h0_ours = h0_torch.clone()

        # Act
        actual_out, actual_hn = rnn_ours(xs_ours, h0_ours)

        # Assert
        torch.testing.assert_close(actual_out, expected_out)
        torch.testing.assert_close(actual_hn, expected_hn)

    def test_when_memory_not_provided_then_defaults_to_zero(self):
        # Arrange
        input_size = 10
        hidden_size = 20
        num_layers = 2
        batch_size = 3
        seq_len = 5
        rnn_torch = nn.RNN(input_size, hidden_size, num_layers)
        torch_parameters = dict(rnn_torch.named_parameters())
        xs_torch = torch.randn(seq_len, batch_size, input_size)
        expected_out, expected_hn = rnn_torch(xs_torch)

        rnn_ours = rnn.RNN(input_size, hidden_size, num_layers)
        with torch.no_grad():
            rnn_ours.cell_layers[0].weight_hh.copy_(torch_parameters['weight_hh_l0'].detach())
            rnn_ours.cell_layers[0].weight_ih.copy_(torch_parameters['weight_ih_l0'].detach())
            rnn_ours.cell_layers[0].bias_ih.copy_(torch_parameters['bias_ih_l0'].detach())
            rnn_ours.cell_layers[0].bias_hh.copy_(torch_parameters['bias_hh_l0'].detach())
            rnn_ours.cell_layers[1].weight_hh.copy_(torch_parameters['weight_hh_l1'].detach())
            rnn_ours.cell_layers[1].weight_ih.copy_(torch_parameters['weight_ih_l1'].detach())
            rnn_ours.cell_layers[1].bias_ih.copy_(torch_parameters['bias_ih_l1'].detach())
            rnn_ours.cell_layers[1].bias_hh.copy_(torch_parameters['bias_hh_l1'].detach())
        xs_ours = xs_torch.clone()

        # Act
        actual_out, actual_hn = rnn_ours(xs_ours)

        # Assert
        torch.testing.assert_close(actual_out, expected_out)
        torch.testing.assert_close(actual_hn, expected_hn)
