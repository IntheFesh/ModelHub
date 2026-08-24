from pathlib import Path

import pytest

from modelhub.data.quick_eval import select_quick_eval_ids
from modelhub.data.safety_adversarial import export_safety_adversarial_set
from modelhub.data.schema import NormalizedSample


def test_select_quick_eval_ids_is_sorted_and_fixed(
    minidev_select_samples: list[NormalizedSample],
) -> None:
    ids = select_quick_eval_ids(minidev_select_samples)
    assert ids == sorted(ids)
    assert len(ids) == len(minidev_select_samples)
    assert ids == select_quick_eval_ids(list(reversed(minidev_select_samples)))


def test_select_quick_eval_ids_rejects_wrong_source(
    minidev_crud_samples: list[NormalizedSample],
) -> None:
    with pytest.raises(ValueError, match="MINIDEV_SELECT"):
        select_quick_eval_ids(minidev_crud_samples)


def test_export_safety_adversarial_set_writes_all_crud_samples(
    minidev_crud_samples: list[NormalizedSample], tmp_path: Path
) -> None:
    out = tmp_path / "safety_adversarial.jsonl"
    export_safety_adversarial_set(minidev_crud_samples, out_path=out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == len(minidev_crud_samples)
    assert all('"expected_gate_outcome": "REJECT"' in line for line in lines)


def test_export_safety_adversarial_set_rejects_wrong_source(
    minidev_select_samples: list[NormalizedSample], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="MINIDEV_CRUD"):
        export_safety_adversarial_set(minidev_select_samples, out_path=tmp_path / "x.jsonl")
