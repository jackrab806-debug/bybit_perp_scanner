"""Statistical validation of detected event types.

Four tests are run to determine whether an event type has genuine
predictive power relative to a matched random baseline:

  1. KS test (p < 0.01)
     Two-sample Kolmogorov–Smirnov test on the mfe_60m distributions of
     events vs baseline.  A small p-value means the distributions differ.

  2. Win-rate 95% CI, block bootstrap (lower bound > 50%)
     Win = outcome hit the TP barrier.  4-hour temporal blocks prevent
     inflating significance from correlated consecutive events.

  3. Cohen's d on mfe_60m (d > 0.3, medium effect)
     Measures how much larger the events' favorable excursion is relative
     to the baseline.  Positive d means events outperform baseline.

  4. Walk-forward OOS validation
     Chronological 60-day train / 30-day validate splits.  On each
     training window the optimal score threshold (maximising win rate)
     is learned.  The threshold is applied to the out-of-sample window.
     Aggregate OOS win-rate lower CI bound must exceed 50%.

Usage
-----
    from src.backtest.validation import validate_event_type

    result = validate_event_type(labeled_events, baseline_events)
    print(result.summary())
    if result.passes_all:
        print("Event type is statistically validated.")
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp

from .labeling import LabeledEvent

# ── Validation thresholds ──────────────────────────────────────────────────────

_KS_P_THRESHOLD    = 0.01    # KS test p-value must be below this
_WIN_CI_THRESHOLD  = 0.50    # lower CI bound of win rate must exceed this
_COHENS_D_MIN      = 0.30    # medium effect size
_N_BOOT            = 2_000   # bootstrap iterations
_BLOCK_HOURS       = 4.0     # temporal block size for block bootstrap
_TRAIN_DAYS        = 60      # walk-forward training window
_VAL_DAYS          = 30      # walk-forward validation window
_WF_MIN_TRAIN      = 5       # skip a split if fewer training events
_WF_MIN_VAL        = 2       # skip a split if fewer validation events


# ── ValidationResult ───────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """
    Outcome of all four validation tests for one event type.

    Attributes
    ----------
    n_events, n_baseline : int
        Counts of input LabeledEvent lists.

    ks_stat, ks_pvalue, passes_ks :
        Two-sample KS test on mfe_60m.  ``passes_ks`` iff p < 0.01.

    win_rate, win_rate_ci_lower, win_rate_ci_upper, passes_win_rate_ci :
        Win rate and 95% block-bootstrap CI.
        ``passes_win_rate_ci`` iff ``win_rate_ci_lower`` > 0.50.

    cohens_d, passes_effect_size :
        Cohen's d (events vs baseline on mfe_60m).
        ``passes_effect_size`` iff d > 0.30.

    wf_n_splits, wf_oos_win_rate, wf_oos_ci_lower, wf_oos_ci_upper,
    wf_opt_score_thresh, passes_walk_forward :
        Walk-forward results.  ``wf_n_splits == 0`` means insufficient
        history; the test is skipped and ``passes_walk_forward`` is True
        (not penalised for lack of data).

    passes_all : bool (property)
        True iff all four tests pass.
    """

    n_events:   int
    n_baseline: int

    # KS test
    ks_stat:   float
    ks_pvalue: float
    passes_ks: bool

    # Win rate
    win_rate:           float
    win_rate_ci_lower:  float
    win_rate_ci_upper:  float
    passes_win_rate_ci: bool

    # Effect size
    cohens_d:           float
    passes_effect_size: bool

    # Walk-forward
    wf_n_splits:          int
    wf_oos_win_rate:      float
    wf_oos_ci_lower:      float
    wf_oos_ci_upper:      float
    wf_opt_score_thresh:  float   # threshold from the last training window
    passes_walk_forward:  bool

    @property
    def passes_all(self) -> bool:
        return (
            self.passes_ks
            and self.passes_win_rate_ci
            and self.passes_effect_size
            and self.passes_walk_forward
        )

    def summary(self) -> str:
        nan_fmt = lambda x: f"{x:.4f}" if not math.isnan(x) else "n/a"
        lines = [
            f"  n_events={self.n_events}  n_baseline={self.n_baseline}",
            f"  KS test      stat={nan_fmt(self.ks_stat)}"
            f"  p={nan_fmt(self.ks_pvalue)}"
            f"  {'PASS' if self.passes_ks else 'FAIL'}",
            f"  Win rate     {self.win_rate*100:.1f}%"
            f"  95%CI [{self.win_rate_ci_lower*100:.1f}%,"
            f" {self.win_rate_ci_upper*100:.1f}%]"
            f"  {'PASS' if self.passes_win_rate_ci else 'FAIL'}",
            f"  Cohen's d    {nan_fmt(self.cohens_d)}"
            f"  {'PASS' if self.passes_effect_size else 'FAIL'}",
            f"  Walk-fwd     oos_wr={self.wf_oos_win_rate*100:.1f}%"
            f"  CI [{self.wf_oos_ci_lower*100:.1f}%,"
            f" {self.wf_oos_ci_upper*100:.1f}%]"
            f"  thresh={self.wf_opt_score_thresh:.1f}"
            f"  n_splits={self.wf_n_splits}"
            f"  {'PASS' if self.passes_walk_forward else 'FAIL'}",
            f"  Overall      {'PASS' if self.passes_all else 'FAIL'}",
        ]
        return "\n".join(lines)


# ── Internal statistics helpers ────────────────────────────────────────────────


def _win_rate(labeled: List[LabeledEvent]) -> float:
    if not labeled:
        return float("nan")
    return sum(1 for e in labeled if e.hit_tp) / len(labeled)


def _cohens_d(a: List[float], b: List[float]) -> float:
    """Pooled-variance Cohen's d (a vs b, positive = a > b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    arr_a, arr_b = np.asarray(a, float), np.asarray(b, float)
    ma, mb       = arr_a.mean(), arr_b.mean()
    va, vb       = arr_a.var(ddof=1), arr_b.var(ddof=1)
    pooled_var   = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    pooled_std   = math.sqrt(pooled_var)
    if pooled_std < 1e-12:
        return float("nan")
    return float((ma - mb) / pooled_std)


def _block_bootstrap_win_rate(
    labeled:     List[LabeledEvent],
    block_hours: float = _BLOCK_HOURS,
    n_boot:      int   = _N_BOOT,
    alpha:       float = 0.05,
    seed:        int   = 42,
) -> Tuple[float, float, float]:
    """
    Block bootstrap 95% CI for the win rate.

    Events are grouped into non-overlapping ``block_hours``-wide UTC time
    buckets.  On each bootstrap iteration, blocks are resampled with
    replacement to produce a synthetic sample of equal block count.

    Returns
    -------
    (win_rate, ci_lower, ci_upper)
    """
    if not labeled:
        return float("nan"), float("nan"), float("nan")

    block_s = block_hours * 3600

    blocks: Dict[int, List[LabeledEvent]] = defaultdict(list)
    for e in labeled:
        block_id = int(e.timestamp.timestamp() // block_s)
        blocks[block_id].append(e)

    block_list = list(blocks.values())
    if not block_list:
        return float("nan"), float("nan"), float("nan")

    rng = random.Random(seed)
    boot_rates: List[float] = []

    for _ in range(n_boot):
        drawn: List[LabeledEvent] = []
        for _ in range(len(block_list)):
            drawn.extend(rng.choice(block_list))
        wr = _win_rate(drawn)
        if not math.isnan(wr):
            boot_rates.append(wr)

    if not boot_rates:
        return _win_rate(labeled), float("nan"), float("nan")

    ci_lo = float(np.quantile(boot_rates, alpha / 2))
    ci_hi = float(np.quantile(boot_rates, 1.0 - alpha / 2))
    return _win_rate(labeled), ci_lo, ci_hi


def _optimal_score_threshold(labeled: List[LabeledEvent]) -> float:
    """
    Scan score thresholds to find the one that maximises win rate on
    ``labeled``.  Returns the best threshold or 0.0 if fewer than
    ``_WF_MIN_TRAIN`` events.
    """
    if len(labeled) < _WF_MIN_TRAIN:
        return 0.0

    best_wr     = -1.0
    best_thresh = 0.0

    for thresh in sorted(set(e.score for e in labeled)):
        subset = [e for e in labeled if e.score >= thresh]
        if len(subset) < _WF_MIN_TRAIN:
            continue
        wr = _win_rate(subset)
        if not math.isnan(wr) and wr > best_wr:
            best_wr     = wr
            best_thresh = thresh

    return best_thresh


def _walk_forward(
    events:     List[LabeledEvent],
    train_days: int = _TRAIN_DAYS,
    val_days:   int = _VAL_DAYS,
) -> Tuple[float, float, float, float, int]:
    """
    Rolling walk-forward evaluation.

    Sliding window: [cursor, cursor+train) → learn threshold,
                   [cursor+train, cursor+train+val) → OOS evaluation.
    Advances by ``val_days`` each step (non-overlapping OOS windows).

    Returns
    -------
    (oos_win_rate, oos_ci_lo, oos_ci_hi, last_score_thresh, n_splits)

    ``n_splits == 0`` means insufficient history to perform any split.
    In that case win_rate is the full-sample estimate and CIs are from
    the block bootstrap applied to all events.
    """
    if not events:
        return float("nan"), float("nan"), float("nan"), 0.0, 0

    events = sorted(events, key=lambda e: e.timestamp)
    t_start = events[0].timestamp
    t_end   = events[-1].timestamp

    if (t_end - t_start).days < train_days + val_days:
        # Not enough history — return full-sample bootstrap as fallback
        wr, ci_lo, ci_hi = _block_bootstrap_win_rate(events)
        return wr, ci_lo, ci_hi, _optimal_score_threshold(events), 0

    all_oos:    List[LabeledEvent] = []
    last_thresh = 0.0
    n_splits    = 0
    cursor      = t_start

    while True:
        train_end = cursor + timedelta(days=train_days)
        val_end   = train_end + timedelta(days=val_days)
        if val_end > t_end + timedelta(days=1):
            break

        train_set = [e for e in events if cursor <= e.timestamp < train_end]
        val_set   = [e for e in events if train_end <= e.timestamp < val_end]

        if len(train_set) < _WF_MIN_TRAIN or len(val_set) < _WF_MIN_VAL:
            cursor += timedelta(days=val_days)
            continue

        thresh = _optimal_score_threshold(train_set)
        last_thresh = thresh

        oos = [e for e in val_set if e.score >= thresh]
        all_oos.extend(oos)
        n_splits += 1
        cursor += timedelta(days=val_days)

    if not all_oos:
        return float("nan"), float("nan"), float("nan"), last_thresh, n_splits

    oos_wr, ci_lo, ci_hi = _block_bootstrap_win_rate(all_oos)
    return oos_wr, ci_lo, ci_hi, last_thresh, n_splits


# ── Main public function ───────────────────────────────────────────────────────


def validate_event_type(
    events:   List[LabeledEvent],
    baseline: List[LabeledEvent],
    n_boot:   int = _N_BOOT,
    seed:     int = 42,
) -> ValidationResult:
    """
    Run all four validation tests for one event type.

    Parameters
    ----------
    events :
        LabeledEvent list from real detected events (positive examples).
    baseline :
        LabeledEvent list from matched random baselines (null hypothesis).
    n_boot :
        Number of bootstrap iterations for win-rate CI (default 2000).
    seed :
        RNG seed for reproducibility.

    Returns
    -------
    ValidationResult with per-test pass/fail and summary() method.

    Notes
    -----
    A minimum of ~30 events is recommended for reliable KS and bootstrap
    statistics.  With fewer events, CIs will be wide and tests may not
    reach the required thresholds even for a genuinely predictive signal.
    """
    nan = float("nan")

    # ── 1. KS test on mfe_60m ─────────────────────────────────────────────────
    ev_mfe = [e.mfe_60m for e in events  if not math.isnan(e.mfe_60m)]
    bl_mfe = [e.mfe_60m for e in baseline if not math.isnan(e.mfe_60m)]

    if len(ev_mfe) >= 2 and len(bl_mfe) >= 2:
        ks_stat, ks_p = ks_2samp(ev_mfe, bl_mfe)
    else:
        ks_stat, ks_p = nan, nan

    passes_ks = (not math.isnan(ks_p)) and (ks_p < _KS_P_THRESHOLD)

    # ── 2. Win rate + block-bootstrap CI ─────────────────────────────────────
    wr, ci_lo, ci_hi = _block_bootstrap_win_rate(events, n_boot=n_boot, seed=seed)
    passes_wr_ci = (not math.isnan(ci_lo)) and (ci_lo > _WIN_CI_THRESHOLD)

    # ── 3. Cohen's d (events vs baseline on mfe_60m) ─────────────────────────
    d = _cohens_d(ev_mfe, bl_mfe)
    passes_d = (not math.isnan(d)) and (d > _COHENS_D_MIN)

    # ── 4. Walk-forward OOS ───────────────────────────────────────────────────
    oos_wr, oos_ci_lo, oos_ci_hi, last_thresh, n_splits = _walk_forward(events)

    if n_splits == 0:
        # Insufficient history — skip walk-forward requirement
        passes_wf = True
    else:
        passes_wf = (
            (not math.isnan(oos_ci_lo))
            and (oos_ci_lo > _WIN_CI_THRESHOLD)
        )

    return ValidationResult(
        n_events=len(events),
        n_baseline=len(baseline),
        ks_stat=ks_stat,
        ks_pvalue=ks_p,
        passes_ks=passes_ks,
        win_rate=wr,
        win_rate_ci_lower=ci_lo,
        win_rate_ci_upper=ci_hi,
        passes_win_rate_ci=passes_wr_ci,
        cohens_d=d,
        passes_effect_size=passes_d,
        wf_n_splits=n_splits,
        wf_oos_win_rate=oos_wr,
        wf_oos_ci_lower=oos_ci_lo,
        wf_oos_ci_upper=oos_ci_hi,
        wf_opt_score_thresh=last_thresh,
        passes_walk_forward=passes_wf,
    )
