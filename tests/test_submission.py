from __future__ import annotations

import numpy as np
import pytest

from astar_island.submission import sanitize_prediction, validate_prediction


def test_sanitize_and_validate():
    p = np.random.rand(4, 5, 6)
    p = sanitize_prediction(p, floor=0.01)
    validate_prediction(p, 4, 5)


def test_validate_wrong_shape():
    p = np.ones((3, 3, 5))
    with pytest.raises(ValueError):
        validate_prediction(p, 3, 3)
