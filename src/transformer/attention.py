"""Scaled dot-product attention implemented with NumPy."""

import numpy as np


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attention outputs and attention weights.

    The last two dimensions of query/key/value are (sequence, features).
    Leading batch dimensions are supported and broadcast according to NumPy.
    A boolean mask uses True for positions that may be attended to; a numeric
    mask is added to the attention scores (for example, 0 and -inf).

    Returns:
        A pair of (output, weights), where weights sum to one over the key
        sequence for every query that has at least one permitted key. An
        entirely masked query produces zero weights and a zero output.
    """
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)

    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must each have at least 2 dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")

    scores = np.matmul(query, np.swapaxes(key, -1, -2))
    scores = scores / np.sqrt(query.shape[-1])

    if mask is not None:
        mask = np.asarray(mask)
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("mask must be broadcastable to attention score shape") from exc
        if mask.dtype == np.bool_:
            scores = np.where(mask, scores, -np.inf)
        else:
            scores = scores + mask

    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exponentials = np.exp(scores - row_max)
    denominator = np.sum(exponentials, axis=-1, keepdims=True)
    weights = np.divide(
        exponentials,
        denominator,
        out=np.zeros_like(exponentials),
        where=denominator != 0,
    )
    output = np.matmul(weights, value)
    return output, weights
