"""Compatibility entry point for the maintained Improved V2 pipeline.

Use ``python -m backend.ai.improved_v2_train`` for the canonical command.
This module remains available for older project instructions without running
training as a side effect of import.
"""

from backend.ai.improved_v2_train import parse_args, train


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args())
