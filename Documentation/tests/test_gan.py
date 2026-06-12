import os
import unittest

import datasets
import numpy as np
import torch

from projects import gan


class TestGetNRealSamples(unittest.TestCase):

    def test_when_n_is_four_then_returns_four_samples(self):
        # Arrange
        n = 4
        dataset_test = datasets.load_from_disk(os.path.join('DATA', 'mnist', 'test'))
        rng = np.random.default_rng(23)
        expected_len_images = n
        expected_len_labels = n
        expected_cls_first_image = torch.FloatTensor
        expected_cls_first_label = int
        expected_max_val_first_image = 1

        # Act
        images, labels = gan.get_n_real_samples(dataset_test, n, rng)
        first_image = images[0]
        first_label = labels[0]
        actual_len_images = len(images)
        actual_len_labels = len(labels)
        actual_max_val_first_image = first_image.max()

        # Assert
        self.assertEqual(actual_len_images, expected_len_images)
        self.assertEqual(actual_len_labels, expected_len_labels)
        self.assertIsInstance(first_image, expected_cls_first_image)
        self.assertIsInstance(first_label, expected_cls_first_label)
        self.assertEqual(actual_max_val_first_image, expected_max_val_first_image)


class TestInit(unittest.TestCase):

    def test_when_called_then_creates_generator_and_discriminator(self):
        # Arrange
        layer_shapes_disc = [4, 3, 2, 1]
        layer_shapes_gen = [4, 3, 2, 4]
        epochs = k = batch_size = seed = 3
        expected_class_generator = gan.Generator
        expected_class_discriminator = gan.Discriminator

        # Act
        model = gan.Gan(layer_shapes_disc, layer_shapes_gen, epochs, k, batch_size, seed)
        actual_discriminator = model.discriminator
        actual_generator = model.generator

        # Assert
        self.assertIsInstance(actual_discriminator, expected_class_discriminator)
        self.assertIsInstance(actual_generator, expected_class_generator)


class TestFit(unittest.TestCase):

    def test_when_called_then_returns_discriminator_and_generator_losses(self):
        # Arrange
        epochs = 3
        k = 2
        batch_size = 3

        layer_shapes_disc = [28 * 28, 3, 2, 1]
        layer_shapes_gen = [4, 3, 2, 28 * 28]
        dataset_test = datasets.load_from_disk(os.path.join('DATA', 'mnist', 'test'))
        seed = 23
        model = gan.Gan(layer_shapes_disc, layer_shapes_gen, epochs, k, batch_size, seed)

        expected_discriminator_losses_shape = (epochs, k)
        expected_generator_losses_shape = (epochs, )

        # Act
        discriminator_losses, generator_losses = model.fit(dataset_test)
        actual_discriminator_losses_shape = discriminator_losses.shape
        actual_generator_losses_shape = generator_losses.shape

        # Assert
        self.assertTupleEqual(actual_discriminator_losses_shape,
                              expected_discriminator_losses_shape)
        self.assertTupleEqual(actual_generator_losses_shape, expected_generator_losses_shape)
