"""The fixed quick-eval subset (CLAUDE.md §7.1).

Quick-eval = Mini-Dev V2's 500 SELECT-only samples. The ID list is fixed
once and written into the dataset manifest; it must never change across
experiments (that's what makes iteration-to-iteration comparisons valid).
"""

from __future__ import annotations

from collections.abc import Sequence

from modelhub.data.schema import NormalizedSample, Source


def select_quick_eval_ids(minidev_select_samples: Sequence[NormalizedSample]) -> list[str]:
    """Return the fixed, sorted quick-eval sample_id list.

    Sorted (not insertion-order) so the list is reproducible regardless of
    how the caller assembled `minidev_select_samples`. Raises `ValueError`
    (a caller-contract violation, not a data-quality event worth a
    ModelHubError/manifest entry) if anything other than
    Source.MINIDEV_SELECT is passed in — mixing sources into the quick-eval
    set would make it non-fixed across dataset versions.
    """
    wrong_source = [s for s in minidev_select_samples if s.source is not Source.MINIDEV_SELECT]
    if wrong_source:
        raise ValueError(
            f"{len(wrong_source)} sample(s) are not Source.MINIDEV_SELECT: "
            f"{[s.sample_id for s in wrong_source][:10]}"
        )
    return sorted(s.sample_id for s in minidev_select_samples)
