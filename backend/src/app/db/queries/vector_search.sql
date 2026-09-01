-- k-NN over chunk embeddings for one tenant.
--
-- The tenant / strategy / model filters sit inside this query, not in
-- application code after the fact: the authorization boundary is the
-- WHERE clause, so a chunk from another tenant can never reach the
-- ranking. `chunks.tenant_id` is denormalised precisely so this filter
-- runs before the ORDER BY rather than after the vector scan.
--
-- Score is cosine similarity (1 - cosine distance); `<=>` is pgvector's
-- cosine-distance operator and matches the HNSW index's
-- `vector_cosine_ops`. Ordering by the raw distance (ascending) is what
-- lets the planner use the index.
SELECT c.id            AS chunk_id,
       c.document_id   AS document_id,
       d.filename      AS filename,
       c.content       AS content,
       c.page_from     AS page_from,
       c.page_to       AS page_to,
       c.section_path  AS section_path,
       c.heading       AS heading,
       1 - (e.vec <=> :qvec) AS score
FROM embeddings e
JOIN chunks c    ON c.id = e.chunk_id
JOIN documents d ON d.id = c.document_id
WHERE c.tenant_id = :tenant_id
  AND c.strategy  = :strategy
  AND e.model     = :model
ORDER BY e.vec <=> :qvec
LIMIT :k;
