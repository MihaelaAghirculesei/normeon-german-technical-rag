from app.domain.models import RetrievedChunk


class NoopReranker:
    """Identity reranker: keeps the incoming order, just truncates to
    `top_k`, scores untouched. The pipeline always calls a reranker, so
    this is what the "reranking off" cell of the experiment matrix uses.
    """

    name = "noop"

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return list(chunks[:top_k])
