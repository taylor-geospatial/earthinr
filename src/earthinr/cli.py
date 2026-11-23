# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Command-line interface for EarthINR."""

import argparse
from collections.abc import Sequence

from . import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Run the EarthINR command-line interface.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog='earthinr',
        description='EarthINR: Implicit Neural Representations for Earth Observation.',
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}'
    )
    parser.parse_args(argv)
    return 0


__all__ = ('main',)
