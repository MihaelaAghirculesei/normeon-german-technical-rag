-- Trigram fallback for code / norm-reference queries ("LH-3.2.1",
-- "UN R79"). The German FTS tokenizer can still fumble an unusual code
-- shape, or the code may be lightly misspelled; word_similarity compares
-- :code against the most similar run of trigrams inside content_norm, so
-- chunk length doesn't drown the score the way plain similarity() would.
--
-- `:code <% content_norm` is the indexable form (idx_chunks_trgm, GIN
-- gin_trgm_ops); its cutoff is pg_trgm.word_similarity_threshold, which
-- the caller SET LOCALs before running this.
SELECT c.id            AS chunk_id,
       c.document_id   AS document_id,
       d.filename      AS filename,
       c.content       AS content,
       c.page_from     AS page_from,
       c.page_to       AS page_to,
       c.section_path  AS section_path,
       c.heading       AS heading,
       d.version_label AS version_label,
       word_similarity(:code, c.content_norm) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.tenant_id = :tenant_id
  AND c.strategy  = :strategy
  AND :code <% c.content_norm
ORDER BY score DESC
LIMIT :k;
