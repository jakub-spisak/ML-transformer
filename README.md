# ML-transformer

A small, readable NumPy project for learning the core pieces behind Transformer models.

## Included

- Scaled dot-product attention with optional boolean or additive masks and batch dimensions.
- Sinusoidal positional encodings.
- Unit tests for masking, batching, numerical stability, and odd feature dimensions.
- GitHub Actions CI on Python 3.10, 3.11, and 3.12.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick example

```python
import numpy as np
from transformer import scaled_dot_product_attention

query = np.array([[1.0, 0.0]])
key = np.array([[1.0, 0.0], [0.0, 1.0]])
value = np.array([[2.0], [8.0]])

# True means the position is available to attention.
output, weights = scaled_dot_product_attention(
    query, key, value, mask=np.array([[True, False]])
)
print(output)  # [[2.]]
print(weights) # [[1. 0.]]
```

This is an educational foundation rather than a production model. A natural next step is to add multi-head attention and a complete encoder block.
