#!/usr/bin/env python3
"""Run the MFAA experiment from a fresh clone without installing the package.

Adds ``src`` to the path, then delegates to ``mfaa.experiment.main``. If you have
installed the package (``pip install -e .``) you can instead run ``mfaa-experiment``
or ``python -m mfaa``.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mfaa.experiment import main  # noqa: E402

if __name__ == "__main__":
    main()
