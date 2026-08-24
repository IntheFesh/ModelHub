from modelhub.data.dedup import dedup_samples
from modelhub.data.schema import NormalizedSample


def test_dedup_drops_question_level_duplicates_with_count(
    bird_train_samples: list[NormalizedSample], bird_dev_samples: list[NormalizedSample]
) -> None:
    # bird_dev's first sample is a question-text duplicate of bird_train's first.
    combined = bird_train_samples + bird_dev_samples
    result = dedup_samples(combined)
    assert result.stats.input_count == len(combined)
    assert result.stats.output_count == len(combined) - 1
    assert result.stats.question_level_duplicates == 1
    assert len(result.dropped_sample_ids) == 1


def test_dedup_first_occurrence_wins(bird_train_samples: list[NormalizedSample]) -> None:
    dup = bird_train_samples[0].model_copy(update={"sample_id": "bird:train:1-dup"})
    result = dedup_samples([bird_train_samples[0], dup])
    assert len(result.samples) == 1
    assert result.samples[0].sample_id == bird_train_samples[0].sample_id


def test_dedup_counts_sql_level_duplicates_within_kept_but_does_not_drop_them() -> None:
    from modelhub.data.schema import Dialect, Source, Split

    a = NormalizedSample(
        sample_id="a",
        db_id="school",
        question="Question A",
        evidence=None,
        gold_sql="SELECT COUNT(*) FROM students",
        difficulty=None,
        source=Source.BIRD,
        split=Split.TRAIN,
        dialect=Dialect.SQLITE,
    )
    b = a.model_copy(update={"sample_id": "b", "question": "Question B (different wording)"})
    result = dedup_samples([a, b])
    assert result.stats.output_count == 2  # both kept
    assert result.stats.sql_level_duplicates_within_kept == 1


def test_dedup_empty_input() -> None:
    result = dedup_samples([])
    assert result.stats.input_count == 0
    assert result.stats.output_count == 0
    assert result.samples == []
