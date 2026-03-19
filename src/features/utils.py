"""Shared statistical utilities for the feature library."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def robust_z(
    value: float,
    series: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-10,
) -> float:
    """Robust z-score using median and median absolute deviation (MAD).

    Unlike the classical z-score, this is resistant to outliers because
    both the centre (median) and scale (MAD) are robust statistics.

    The scale factor 1.4826 makes MAD asymptotically consistent with the
    standard deviation for normally distributed data.

    Parameters
    ----------
    value:
        The observation to score.
    series:
        Reference distribution (NaN values are silently dropped).
    epsilon:
        Floor added to the denominator to avoid division by zero when
        all values in the series are identical.

    Returns
    -------
    float – NaN when fewer than 2 non-NaN observations are available.
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-6:
        # MAD is zero — all values nearly identical; fall back to std
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1e-6
        if std < 1e-6:
            return 0.0  # no variation at all
        z = (value - med) / std
    else:
        z = (value - med) / (1.4826 * mad + epsilon)
    return float(max(-10.0, min(10.0, z)))
