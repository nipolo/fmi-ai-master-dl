import unittest

import torch
from torch import nn

from projects import gan


class TestInit(unittest.TestCase):

    def test_when_called_then_builds_linear_model(self):
        # Arrange
        expected_num_in = 28 * 28
        expected_num_out = 1
        expected_cls_last_layer = nn.Sigmoid
        expected_cls_layers = nn.ModuleList
        expected_cls_disc = nn.Module
        layer_neurons = [expected_num_in, 5, expected_num_out]

        # Act
        discriminator = gan.Discriminator(layer_neurons)
        actual_num_in = discriminator.layers[0].in_features
        actual_num_out = discriminator.layers[-2].out_features
        actual_last_layer = discriminator.layers[-1]
        actual_layers = discriminator.layers

        # Assert
        self.assertEqual(actual_num_in, expected_num_in)
        self.assertEqual(actual_num_out, expected_num_out)
        self.assertIsInstance(actual_last_layer, expected_cls_last_layer)
        self.assertIsInstance(actual_layers, expected_cls_layers)
        self.assertIsInstance(discriminator, expected_cls_disc)


class TestForward(unittest.TestCase):

    def test_when_called_then_returns_predictions(self):
        # Arrange
        batch_size = 5
        expected_num_out = 1
        random_noise = torch.rand(batch_size, 28 * 28)
        layer_neurons = [28 * 28, 5, expected_num_out]
        expected_shape_output = (batch_size, expected_num_out)

        # Act
        discriminator = gan.Discriminator(layer_neurons)
        actual_shape_output = discriminator(random_noise).shape

        # Assert
        self.assertEqual(actual_shape_output, expected_shape_output)
