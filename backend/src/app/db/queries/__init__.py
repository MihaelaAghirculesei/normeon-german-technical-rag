"""Hand-written SQL that's clearer as SQL than as a query builder lives in
`.sql` files next to this module; `load_sql` reads one by stem.

Kept as files (not inline strings) so they stay syntax-highlighted,
greppable, and reviewable on their own. Cached because the set is small
and fixed for the life of the process.
"""

from functools import lru_cache
from pathlib import Path

_QUERIES_DIR = Path(__file__).parent


@lru_cache
def load_sql(name: str) -> str:
    return (_QUERIES_DIR / f"{name}.sql").read_text(encoding="utf-8")
