from dataclasses import replace

from sentence_transformers import CrossEncoder

from app.domain.models import RetrievedChunk


class CrossEncoderReranker:
    """A sentence-transformers `CrossEncoder` (default bge-reranker-v2-m3)
    scoring `(query, chunk.content)` pairs. The model is loaded lazily on
    the first `rerank` call so constructing this at DI time stays cheap
    and importing the app never pulls the weights.
    """

    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def _loaded_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._loaded_model.predict([(query, chunk.content) for chunk in chunks])
        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:top_k]]
