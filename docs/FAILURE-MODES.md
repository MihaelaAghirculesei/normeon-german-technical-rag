# Failure modes

An honest catalogue of where this system is known to be weak, why, and
what (if anything) compensates. Grows as each stage is built and measured;
the evaluation in Week 4 will attach numbers to several of these.

## Full-text search

### German compound nouns are not decomposed

PostgreSQL's `german` snowball stemmer stems but does not split compounds.
`Fahrzeugzulassung` becomes the lexeme `fahrzeugzulass`, which does **not**
match a query for `Zulassung` (`zulass`):

```sql
SELECT to_tsvector('german', 'Fahrzeugzulassung')
       @@ websearch_to_tsquery('german', 'Zulassung');   -- => false
```

German legal and technical text is full of these (`Betriebserlaubnis`,
`Hauptuntersuchung`, `Rückhalteeinrichtung`, `Beleuchtungseinrichtung`),
so a full-text query phrased with a different compounding than the source
text silently returns nothing.

**Not fixed.** Third-party decompounders (e.g. a CompoundWordTokenFilter
port, or a dictionary splitter) were considered and rejected for this
project: they need a maintained German dictionary, misfire on domain
terms, and add a dependency whose failures are hard to see. Recognising
and measuring the limit is worth more here than papering over it.

**Mitigation.** The vector branch is not lexical and handles compound
paraphrase well (verified in `scripts/try_fts_search.py`: "Wie hell
müssen die Frontleuchten strahlen?" returns nothing from FTS but the
right `§ 50` chunk from vector). Day 8's RRF fusion runs both branches so
the vector side covers the FTS side's blind spot and vice versa.

### tsquery AND semantics punish paraphrased questions

`websearch_to_tsquery` ANDs the query terms. A natural-language question
("Wann muss ein Auto zum TÜV?") contributes terms the corpus never uses
verbatim ("Auto", "TÜV" vs. the statute's "Kraftfahrzeug",
"Hauptuntersuchung"), and one missing term drops the whole match. FTS
here is a keyword / code-reference tool, not a question-answering tool.

**Mitigation.** Same as above — the vector branch, and hybrid fusion on
Day 8. `normalize_de` canonicalises requirement codes and norm references
(`LH-3.2.1`, `UN R79`) into single tokens precisely so the one thing FTS
*should* be reliable at — exact reference lookup — is not defeated by
tokenisation.

## Parsing / chunking

Carried over from earlier days, revisit if Week 4 eval shows they matter:

- The numbered-heading heuristic false-positives on numeric data inside
  annex tables that `find_tables()` misses (Day 3 notes).
- Cascading merges of tiny table-of-contents lines produce one large
  low-value chunk; a section just over the split threshold can leave a
  sub-100-token runt piece (Day 4 notes).
- No real character offsets into the source PDF yet (`char_start`/
  `char_end` are `0`/`len`), so highlight-in-original is not possible
  until that is tracked.
