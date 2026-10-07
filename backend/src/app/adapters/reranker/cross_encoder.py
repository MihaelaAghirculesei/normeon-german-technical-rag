import threading
from dataclasses import replace

from sentence_transformers import CrossEncoder

from app.domain.models import RetrievedChunk


class CrossEncoderReranker:
    """A sentence-transformers `CrossEncoder` (default bge-reranker-v2-m3)
    scoring `(query, chunk.content)` pairs. The model is loaded lazily on
    the first `rerank` call so constructing this at DI time stays cheap
    and importing the app never pulls the weights.

    `max_length` caps the tokens of one pair; the tail of a longer chunk
    is truncated before scoring. `None` leaves the tokenizer's own limit
    in place (8192 for bge-reranker-v2-m3), which scores every chunk in
    full at a cost that grows with its length.
    """

    name = "cross_encoder"

    def __init__(self, model_name: str, max_length: int | None = None) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model: CrossEncoder | None = None
        # Same race as LocalE5Embedder's _loaded_model (see its comment):
        # `rerank` runs off-thread and several questions run concurrently
        # under the eval runner, so an unlocked lazy-load can have two
        # threads constructing a CrossEncoder at once.
        self._load_lock = threading.Lock()

    @property
    def _loaded_model(self) -> CrossEncoder:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    # only pass a cap when there is one: `max_length=None`
                    # is the library's own "use the tokenizer limit" default,
                    # but its signature is typed `int`
                    if self._max_length is None:
                        self._model = CrossEncoder(self._model_name)
                    else:
                        self._model = CrossEncoder(self._model_name, max_length=self._max_length)
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
