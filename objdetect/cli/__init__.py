"""Command-line entry points for the object-detection project.

Each module here is a thin ``argparse`` wrapper around the library packages
(``data``, ``eda``, ``models``, ``training``, ``evaluation``) and is run with
``python -m objdetect.cli.<name>`` — for example
``python -m objdetect.cli.train --help``.
"""
