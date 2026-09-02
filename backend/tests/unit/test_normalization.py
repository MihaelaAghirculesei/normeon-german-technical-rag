"""Pure tests for normalize_de / extract_code.

The umlaut-folding assertions here document *why* normalize_de doesn't
transliterate: PostgreSQL's `german` config already does it. Those live
in the integration suite (they need a database); this file only covers
the string rewriting normalize_de does itself.
"""

import unicodedata

import pytest

from app.domain.normalization import extract_code, normalize_de


@pytest.mark.parametrize("written", ["LH 3.2.1", "LH-3.2.1", "LH3.2.1"])
def test_requirement_code_spellings_collapse_to_one_token(written: str) -> None:
    assert normalize_de(written) == "lh3.2.1"


def test_requirement_code_inside_a_sentence() -> None:
    assert normalize_de("Welche Anforderungen stellt LH-3.2.1 an die Lenkung?") == (
        "welche anforderungen stellt lh3.2.1 an die lenkung?"
    )


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("UN R79", "unr79"),
        ("UN-R79", "unr79"),
        ("ECE R79", "ecer79"),
        ("ECE-R79", "ecer79"),
        ("R 79", "r79"),
        ("R-79", "r79"),
        ("R79", "r79"),
        ("R 79H", "r79h"),
    ],
)
def test_norm_references_collapse_to_one_token(written: str, expected: str) -> None:
    assert normalize_de(written) == expected


def test_ordinary_lowercase_word_before_a_number_is_not_treated_as_a_code() -> None:
    # "bis" is a common word, not a code prefix -- must stay two tokens.
    assert normalize_de("gilt bis 3.2.1 der Norm") == "gilt bis 3.2.1 der norm"


def test_word_containing_r_before_a_number_is_left_alone() -> None:
    assert normalize_de("in der 3 Zeilen langen Tabelle") == "in der 3 zeilen langen tabelle"


def test_whitespace_and_newlines_are_collapsed() -> None:
    assert normalize_de("Zeile eins\n\n  Zeile   zwei\t") == "zeile eins zeile zwei"


def test_umlauts_are_preserved_not_transliterated() -> None:
    # The `german` FTS config folds ü/ue itself; normalize_de must not
    # pre-empt it, or the two branches would disagree.
    assert normalize_de("Prüfung der Tür") == "prüfung der tür"


def test_input_is_nfc_normalised() -> None:
    decomposed = unicodedata.normalize("NFD", "Prüfung")
    assert decomposed != "Prüfung"  # sanity: it really is decomposed
    assert normalize_de(decomposed) == "prüfung"


def test_normalize_de_is_idempotent() -> None:
    messy = "Nach LH-3.2.1 und UN R79 gilt für die Prüfung\nfolgender Wert."
    once = normalize_de(messy)
    assert normalize_de(once) == once


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Welche Anforderungen stellt LH-3.2.1?", "lh3.2.1"),
        ("Was verlangt die Regelung R 79 hier?", "r79"),
        ("Gilt UN-R79 für Anhänger?", "unr79"),
        ("Wie hoch ist die zulässige Lenkkraft?", None),
        ("Was bedeutet Betriebserlaubnis?", None),
    ],
)
def test_extract_code_picks_out_code_shaped_tokens(question: str, expected: str | None) -> None:
    assert extract_code(question) == expected
