"""Hand-written SQL text and the row -> `RetrievedChunk` mapping shared by
the retrieval branches. Kept in one place so the three branches stay in
sync on the column list they SELECT.
"""

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from app.db.models import EMBEDDING_DIM
from app.db.queries import load_sql
from app.domain.models import RetrievedChunk

VECTOR_SEARCH_SQL = text(load_sql("vector_search")).bindparams(
    bindparam("qvec", type_=Vector(EMBEDDING_DIM)),
)
FTS_SEARCH_SQL = text(load_sql("fts_search"))
TRGM_SEARCH_SQL = text(load_sql("trgm_search"))


def row_to_chunk(row: Any) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        filename=row.filename,
        content=row.content,
        page_from=row.page_from,
        page_to=row.page_to,
        section_path=row.section_path,
        heading=row.heading,
        score=float(row.score),
        version_label=row.version_label,
    )
