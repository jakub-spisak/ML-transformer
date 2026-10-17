"""Sinusoidal positional encodings from the original Transformer."""

import numpy as np


def sinusoidal_positional_encoding(
    sequence_length: int,
    model_dimension: int,
) -> np.ndarray:
    """Return positional encodings with shape (sequence_length, model_dimension).

    Even channels use sine and odd channels use cosine. The implementation
    also supports odd model dimensions.
    """
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if model_dimension <= 0:
        raise ValueError("model_dimension must be positive")

    positions = np.arange(sequence_length, dtype=np.float64)[:, np.newaxis]
    channel_pairs = np.arange(0, model_dimension, 2, dtype=np.float64)
    angle_rates = np.exp(-np.log(10000.0) * channel_pairs / model_dimension)
    angles = positions * angle_rates

    encoding = np.zeros((sequence_length, model_dimension), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles[:, : model_dimension // 2])
    return encoding
