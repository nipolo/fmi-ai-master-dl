import os
import pathlib as pl
import unittest

import datasets
import numpy as np

from projects import gan, main


class TestPlotAndSaveToDisk(unittest.TestCase):

    def test_when_called_then_creates_file(self):
        # Arrange
        dataset = datasets.load_from_disk(os.path.join('DATA', 'mnist', 'test'))
        rng = np.random.default_rng(42)
        images, labels = gan.get_n_real_samples(dataset, 4, rng)
        filename = pl.Path('real_samples.png')
        nrows = 2
        ncols = 2

        # Act
        main.plot_and_save_to_disk(nrows, ncols, str(filename), images, labels)

        # Assert
        self.assertTrue(filename.exists())

        # Clean
        filename.unlink()


class TestRun(unittest.TestCase):

    def test_when_no_errors_then_returns_zero(self):
        # Arrange
        expected = 0

        # Act
        actual = main.run()

        # Assert
        self.assertEqual(actual, expected)

        # Clean
        pl.Path('images_real.png').unlink()
