from modelhub.data.hashing import dataset_content_hash, question_hash, sql_hash
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split


def _sample(sample_id: str) -> NormalizedSample:
    return NormalizedSample(
        sample_id=sample_id,
        db_id="school",
        question="How many students?",
        evidence=None,
        gold_sql="SELECT COUNT(*) FROM students",
        difficulty=None,
        source=Source.BIRD,
        split=Split.TRAIN,
        dialect=Dialect.SQLITE,
    )


def test_question_hash_ignores_incidental_whitespace_and_case() -> None:
    assert question_hash("How many  students?") == question_hash("how many students?  ")


def test_question_hash_distinguishes_different_questions() -> None:
    assert question_hash("How many students?") != question_hash("How many teachers?")


def test_sql_hash_ignores_incidental_whitespace_and_case() -> None:
    assert sql_hash("SELECT * FROM t") == sql_hash("select   *  from t")


def test_dataset_content_hash_order_independent() -> None:
    samples = [_sample("a"), _sample("b")]
    assert dataset_content_hash(samples) == dataset_content_hash(list(reversed(samples)))


def test_dataset_content_hash_changes_with_content() -> None:
    a = [_sample("a")]
    b = [_sample("a").model_copy(update={"gold_sql": "SELECT 1"})]
    assert dataset_content_hash(a) != dataset_content_hash(b)


def test_dataset_content_hash_format() -> None:
    assert dataset_content_hash([_sample("a")]).startswith("sha256:")
