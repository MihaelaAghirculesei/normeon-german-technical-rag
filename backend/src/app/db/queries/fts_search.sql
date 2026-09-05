-- German full-text search over the `tsv` generated column
-- (to_tsvector('german', content_norm)), tenant-filtered in the WHERE so
-- the authorization boundary is the query, same as vector_search.sql.
--
-- :q is already run through normalize_de by the caller, so the code /
-- norm-reference collapsing that happened at ingestion happens here too.
-- websearch_to_tsquery keeps a lay user's quotes and `-word` working;
-- ts_rank_cd rewards matches whose terms sit close together.
SELECT c.id            AS chunk_id,
       c.document_id   AS document_id,
       d.filename      AS filename,
       c.content       AS content,
       c.page_from     AS page_from,
       c.page_to       AS page_to,
       c.section_path  AS section_path,
       c.heading       AS heading,
       d.version_label AS version_label,
       ts_rank_cd(c.tsv, websearch_to_tsquery('german', :q)) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.tenant_id = :tenant_id
  AND c.strategy  = :strategy
  AND c.tsv @@ websearch_to_tsquery('german', :q)
ORDER BY score DESC
LIMIT :k;
