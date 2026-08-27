from sentence_transformers import SentenceTransformer

from app.adapters.embedding.base import Vector, add_passage_prefix, add_query_prefix


class LocalE5Embedder:
    """Runs a multilingual-e5 model in-process via sentence-transformers.
    The model is loaded lazily on first use so simply constructing this
    class (e.g. at DI time) never triggers the multi-GB download/load."""

    name = "e5_local"

    def __init__(self, model_name: str, dim: int) -> None:
        self.dim = dim
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def _loaded_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_passages(self, texts: list[str]) -> list[Vector]:
        prefixed = [add_passage_prefix(t) for t in texts]
        vectors = self._loaded_model.encode(prefixed, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> Vector:
        vector = self._loaded_model.encode(add_query_prefix(text), normalize_embeddings=True)
        result: Vector = vector.tolist()
        return result
