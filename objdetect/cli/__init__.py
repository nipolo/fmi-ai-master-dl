"""Command-line entry points for the object-detection project.

Each module here is a thin ``argparse`` wrapper around the library packages
(``data``, ``eda``, ``models``, ``training``, ``evaluation``). Entry points are
grouped by purpose into subpackages and run with
``python -m objdetect.cli.<group>.<name>``:

- ``data``       — ``download_data``, ``prepare_cone_dataset``, ``prepare_cone_coco``
- ``training``   — ``train``, ``train_cone_frcnn``, ``train_cone_yolo``, ``run_experiments``
- ``evaluation`` — ``evaluate``, ``benchmark_baselines``
- ``figures``    — ``plot_lr_schedules``, ``compare_models_visual``

For example: ``python -m objdetect.cli.training.train --help``.
"""
