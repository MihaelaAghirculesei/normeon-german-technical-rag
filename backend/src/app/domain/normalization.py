"""German technical-text normalisation for full-text search.

`normalize_de` is applied identically to two things:
- chunk `content` on ingestion, feeding the `tsv` generated column
  (`to_tsvector('german', content_norm)`);
- the raw question before it becomes a `tsquery`.

Both sides going through the same function is the point -- a requirement
code written three different ways in a document and a fourth way in a
query all collapse to one token.
"""

import re
import unicodedata

# A requirement code: an all-caps 2-4 letter prefix followed by a dotted,
# hierarchical number -- "LH 3.2.1", "LH-3.2.1", "LH3.2.1". The number
# must contain at least one dot; that (and the all-caps prefix) keeps
# ordinary prose like "bis 3.2.1" or "Abschnitt 4" from matching.
_REQUIREMENT_CODE = re.compile(r"\b([A-ZÄÖÜ]{2,4})[ \-]?(\d+(?:\.\d+)+)\b")

# Norm references. "UN R79" / "UN-R79" / "ECE R79" / "ECE-R79" -> "unr79" /
# "ecer79"; a bare "R 79" / "R-79" -> "r79". Without this `to_tsvector`
# splits "R 79" into the tokens `r` and `79` and the reference can't be
# searched for as a unit.
_UN_ECE_NORM = re.compile(r"\b(UN|ECE)[ \-]?R[ \-]?(\d+[A-Za-z]?)\b", re.IGNORECASE)
_BARE_R_NORM = re.compile(r"\bR[ \-]?(\d+[A-Za-z]?)\b", re.IGNORECASE)

_WHITESPACE = re.compile(r"\s+")


def normalize_de(text: str) -> str:
    """Normalise German technical text for full-text indexing and querying.

    - Unicode NFC, then lower-cased.
    - Requirement codes collapsed to one token: `LH 3.2.1` / `LH-3.2.1` /
      `LH3.2.1` -> `lh3.2.1`.
    - Norm references collapsed to one token: `UN R79` / `ECE-R79` ->
      `unr79` / `ecer79`; `R 79` -> `r79`.
    - Whitespace (including newlines) collapsed to single spaces.

    Deliberately *not* done:
    - Umlaut folding / transliteration. PostgreSQL's `german` config
      already maps `ü`/`ue`, `ö`/`oe`, `ä`/`ae`, `ß`/`ss` onto the same
      lexeme (`Prüfung` and `Pruefung` both stem to `prufung`), so doing
      it here would only add noise -- covered by test_normalization.py.
    - German compound splitting -- the `german` snowball stemmer does not
      decompose (`Fahrzeugzulassung` does not match `Zulassung`); see
      docs/FAILURE-MODES.md.

    Idempotent: `normalize_de(normalize_de(x)) == normalize_de(x)`.
    """
    text = unicodedata.normalize("NFC", text)
    text = _REQUIREMENT_CODE.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)
    text = _UN_ECE_NORM.sub(lambda m: f"{m.group(1)}r{m.group(2)}", text)
    text = _BARE_R_NORM.sub(lambda m: f"r{m.group(1)}", text)
    text = text.lower()
    return _WHITESPACE.sub(" ", text).strip()


# Query shapes worth a trigram fallback on top of the tsquery branch: an
# all-caps prefix (one group, or two like "UN R"), an optional separator,
# then a number that may be dotted or carry a trailing letter -- "LH-3.2.1",
# "UN R79", "R 79", "R79H". All-caps only, so ordinary capitalised words
# ("Der 3. Absatz") don't register as codes.
_CODE_SHAPED = re.compile(
    r"\b[A-ZÄÖÜ]{1,4}(?:[ \-]?[A-ZÄÖÜ]{1,3})?[ \-]?\d+(?:\.\d+)*[A-Za-z]?\b"
)


def extract_code(question: str) -> str | None:
    """If the question contains a code/norm-shaped token, return it in the
    same canonical form `normalize_de` would give it, otherwise None. Used
    to decide whether to run the trigram fallback branch and what to match
    against.
    """
    match = _CODE_SHAPED.search(question)
    if match is None:
        return None
    return normalize_de(match.group(0))
