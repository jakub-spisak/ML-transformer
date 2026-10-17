import numpy as np
import pytest

from transformer import scaled_dot_product_attention, sinusoidal_positional_encoding


def test_attention_returns_expected_single_key_result():
    query = np.array([[1.0, 0.0]])
    key = np.array([[1.0, 1.0]])
    value = np.array([[3.0, 4.0]])

    output, weights = scaled_dot_product_attention(query, key, value)

    np.testing.assert_allclose(output, [[3.0, 4.0]])
    np.testing.assert_allclose(weights, [[1.0]])


def test_attention_weights_sum_to_one_and_respect_mask():
    query = np.array([[1.0, 0.0]])
    key = np.array([[1.0, 0.0], [0.0, 1.0]])
    value = np.array([[2.0], [8.0]])

    output, weights = scaled_dot_product_attention(
        query, key, value, mask=np.array([[True, False]])
    )

    np.testing.assert_allclose(weights, [[1.0, 0.0]])
    np.testing.assert_allclose(output, [[2.0]])


def test_fully_masked_query_produces_zeros():
    q = np.ones((1, 2))
    k = np.ones((2, 2))
    v = np.array([[2.0], [5.0]])

    output, weights = scaled_dot_product_attention(q, k, v, mask=False)

    np.testing.assert_array_equal(weights, [[0.0, 0.0]])
    np.testing.assert_array_equal(output, [[0.0]])


def test_attention_supports_batches_and_stable_large_scores():
    q = np.full((2, 1, 4), 1e100)
    k = np.full((2, 3, 4), 1e100)
    v = np.arange(6.0).reshape(2, 3, 1)

    output, weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (2, 1, 1)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)


@pytest.mark.parametrize(
    ("length", "dimension"),
    [(0, 4), (3, 5), (5, 6)],
)
def test_positional_encoding_shape_and_values(length, dimension):
    encoding = sinusoidal_positional_encoding(length, dimension)

    assert encoding.shape == (length, dimension)
    if length:
        np.testing.assert_allclose(encoding[0, 0::2], 0.0)
        np.testing.assert_allclose(encoding[0, 1::2], 1.0)


def test_positional_encoding_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        sinusoidal_positional_encoding(2, 0)
