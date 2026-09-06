"""Pure unit tests for domain/config_hash.py."""

from app.domain.config_hash import compute_config_hash


def test_same_fields_hash_identically_regardless_of_kwarg_order() -> None:
    a = compute_config_hash(reranker_provider="cross_encoder", rrf_k=60)
    b = compute_config_hash(rrf_k=60, reranker_provider="cross_encoder")

    assert a == b


def test_a_different_value_changes_the_hash() -> None:
    a = compute_config_hash(rrf_k=60)
    b = compute_config_hash(rrf_k=61)

    assert a != b


def test_a_different_field_set_changes_the_hash() -> None:
    a = compute_config_hash(rrf_k=60)
    b = compute_config_hash(rrf_k=60, reranker_provider="noop")

    assert a != b


def test_output_is_a_64_char_hex_sha256() -> None:
    digest = compute_config_hash(a=1)

    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_is_deterministic_across_calls() -> None:
    assert compute_config_hash(x="y", n=1) == compute_config_hash(x="y", n=1)


def test_no_fields_still_produces_a_stable_hash() -> None:
    assert compute_config_hash() == compute_config_hash()
