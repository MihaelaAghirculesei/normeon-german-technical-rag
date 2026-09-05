"""Ad-hoc check for Day 14: hit a *running* backend's streaming endpoint
and print each SSE event as it arrives, timestamped from the start of the
request -- the scriptable cousin of the plan's `curl -N` acceptance
check (`docs/adr` aside, this project prefers a script over a shell
one-liner so it survives as something re-runnable).

Needs the backend actually running and reachable, unlike try_chat.py
(which calls generate_answer in-process): either

    docker compose up -d

or a local `uvicorn app.main:app --port 8010` from backend/. Default URL
matches this machine's docker-compose port remap (see CLAUDE.md
"Environment specifics"); override with --url or $BACKEND_URL.

Run:
    cd backend
    .venv/Scripts/python scripts/try_chat_stream.py \\
        "Welche Vorschriften gelten fuer Scheinwerfer?"

Fatto quando (plan, Giorno 14): the events print in order -- stage
(retrieving/reranking) -> sources -> stage(generating) -> token* -> done.
Ctrl-C mid-stream and check the backend's own logs for a
`client_disconnected` event (the other half of the acceptance bar; this
script only demonstrates the client side of that).
"""

import argparse
import os
import sys
import time

import httpx

from app.db.seed import DEMO_TENANT_ID

DEFAULT_URL = "http://localhost:8010"


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--url", default=os.environ.get("BACKEND_URL", DEFAULT_URL))
    parser.add_argument("--strategy", choices=["fixed_500", "structural"], default=None)
    args = parser.parse_args()

    payload = {"question": args.question, "tenant_id": str(DEMO_TENANT_ID)}
    if args.strategy:
        payload["strategy"] = args.strategy

    started = time.perf_counter()
    with (
        httpx.Client(timeout=None) as client,
        client.stream("POST", f"{args.url}/api/v1/chat/stream", json=payload) as response,
    ):
        print(f"POST {args.url}/api/v1/chat/stream -> {response.status_code}\n")
        for line in response.iter_lines():
            if not line:
                continue
            print(f"[{time.perf_counter() - started:6.2f}s] {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
