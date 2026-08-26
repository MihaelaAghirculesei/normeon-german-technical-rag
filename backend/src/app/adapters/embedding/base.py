from typing import Protocol

Vector = list[float]


class EmbeddingAdapter(Protocol):
    """Separate `embed_passages`/`embed_query` methods on purpose: e5 models
    require a `passage: `/`query: ` prefix depending on which side of
    retrieval the text is on, and mixing them up silently costs ~10 recall
    points. A single `embed(texts, mode=...)` method makes that mistake one
    forgotten argument away; two methods make it a type error instead."""

    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[Vector]: ...
    def embed_query(self, text: str) -> Vector: ...


def add_passage_prefix(text: str) -> str:
    return f"passage: {text}"


def add_query_prefix(text: str) -> str:
    return f"query: {text}"
