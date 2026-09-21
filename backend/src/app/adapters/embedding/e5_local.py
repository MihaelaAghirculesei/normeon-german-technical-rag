import threading

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
        # `embed_query`/`embed_passages` run off-thread (`asyncio.to_
        # thread`), and Giorno 17-18's eval runner drives several
        # questions concurrently (its own semaphore, default 4) -- with
        # no lock here, two threads could both see `_model is None` and
        # each start constructing a SentenceTransformer at the same
        # time, racing on transformers/accelerate's meta-device init and
        # surfacing as "Cannot copy out of meta tensor; no data!" instead
        # of a clean load. Real failure, reproduced running the real
        # 50-question eval set for the first time (prior days only ever
        # drove this with a fake/hashing embedder in tests, or a single
        # request at a time from scripts).
        self._load_lock = threading.Lock()

    @property
    def _loaded_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._load_lock:
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
