"""
cohere_client.py — thin wrapper over Cohere Embed + Rerank.

Design goal: identical interface whether running against the real Cohere API
or a deterministic local mock. Set CO_API_KEY in .env to use the real API.
Without a key, a mock client runs so all logic can be developed and tested
offline without spending trial quota.

Rate-limit aware: embeddings AND rerank results are cached to disk, and real
API calls are throttled to stay under the trial key's 10 calls/minute limit.
"""

import os
import json
import hashlib
import math
import time
import threading
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv optional in mock mode

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
EMBED_CACHE = CACHE_DIR / "embed_cache.json"
RERANK_CACHE = CACHE_DIR / "rerank_cache.json"
CHAT_CACHE = CACHE_DIR / "chat_cache.json"

# Trial keys allow 10 calls/min. We pace real calls to ~1 every 7s = ~8.5/min,
# comfortably under the cap with margin for clock jitter.
_MIN_SECONDS_BETWEEN_CALLS = 7.0
_last_call_time = [0.0]
_throttle_lock = threading.Lock()


def _throttle():
    with _throttle_lock:
        elapsed = time.time() - _last_call_time[0]
        wait = _MIN_SECONDS_BETWEEN_CALLS - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.time()


def _load_cache():
    if EMBED_CACHE.exists():
        return json.loads(EMBED_CACHE.read_text())
    return {}


def _save_cache(cache):
    EMBED_CACHE.write_text(json.dumps(cache))


def _load_rerank_cache():
    if RERANK_CACHE.exists():
        return json.loads(RERANK_CACHE.read_text())
    return {}


def _save_rerank_cache(cache):
    RERANK_CACHE.write_text(json.dumps(cache))


def _load_chat_cache():
    if CHAT_CACHE.exists():
        return json.loads(CHAT_CACHE.read_text())
    return {}


def _save_chat_cache(cache):
    CHAT_CACHE.write_text(json.dumps(cache))


def _hash(model, text):
    return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Mock implementations — deterministic, no network, no quota.
# The mock is intentionally "good enough to be realistic": embeddings are a
# bag-of-words hashed vector so semantically similar text scores higher, and
# rerank reuses cosine similarity. This lets the ratchet loop produce real,
# non-random score movement offline.
# ----------------------------------------------------------------------------
_VOCAB_DIM = 256


def _mock_embed_one(text):
    vec = [0.0] * _VOCAB_DIM
    for tok in text.lower().split():
        tok = "".join(c for c in tok if c.isalnum())
        if not tok:
            continue
        idx = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % _VOCAB_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class CohereClient:
    def __init__(self):
        self.api_key = os.environ.get("CO_API_KEY")
        self.mode = "real" if self.api_key else "mock"
        self._cache = _load_cache()
        self._rerank_cache = _load_rerank_cache()
        self._chat_cache = _load_chat_cache()
        self._client = None
        if self.mode == "real":
            import cohere  # imported lazily so mock mode needs no install
            self._client = cohere.ClientV2(self.api_key)

    # -- Embed -------------------------------------------------------------
    def embed(self, texts, model="embed-v4.0", input_type="search_document"):
        """Return one vector per input text. Cached by (model, text)."""
        results = [None] * len(texts)
        to_fetch, fetch_idx = [], []
        for i, t in enumerate(texts):
            key = _hash(model + input_type, t)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_fetch.append(t)
                fetch_idx.append((i, key))

        if to_fetch:
            if self.mode == "mock":
                vecs = [_mock_embed_one(t) for t in to_fetch]
            else:
                _throttle()
                resp = self._client.embed(
                    texts=to_fetch,
                    model=model,
                    input_type=input_type,
                    embedding_types=["float"],
                )
                vecs = resp.embeddings.float
            for (i, key), v in zip(fetch_idx, vecs):
                results[i] = v
                self._cache[key] = v
            _save_cache(self._cache)
        return results

    # -- Rerank ------------------------------------------------------------
    def rerank(self, query, documents, model="rerank-v3.5", top_n=None):
        """Return list of {index, relevance_score} sorted best-first.
        Cached by (model, query, documents) so re-runs cost no API calls."""
        if self.mode == "mock":
            qv = _mock_embed_one(query)
            scored = []
            for i, d in enumerate(documents):
                dv = _mock_embed_one(d)
                scored.append({"index": i, "relevance_score": max(0.0, _cosine(qv, dv))})
            scored.sort(key=lambda x: x["relevance_score"], reverse=True)
            return scored[: top_n or len(documents)]

        # real mode: check cache first
        cache_key = _hash(model + query, "||".join(documents))
        if cache_key in self._rerank_cache:
            return self._rerank_cache[cache_key]

        _throttle()
        resp = self._client.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n or len(documents),
        )
        result = [
            {"index": r.index, "relevance_score": r.relevance_score}
            for r in resp.results
        ]
        self._rerank_cache[cache_key] = result
        _save_rerank_cache(self._rerank_cache)
        return result

    # -- Chat / generation -------------------------------------------------
    def generate(self, prompt, model="command-r-plus-08-2024", max_tokens=300):
        """Generate an answer from a prompt. Cached by (model, prompt).
        In mock mode, returns the context block verbatim so that answer
        correctness depends entirely on whether the right chunks were
        retrieved — retrieval quality flows through to answer quality."""
        if self.mode == "mock":
            return self._mock_generate(prompt)

        cache_key = _hash(model + "chat", prompt)
        if cache_key in self._chat_cache:
            return self._chat_cache[cache_key]

        _throttle()
        try:
            resp = self._client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.message.content[0].text
        except Exception as e:
            # Graceful degradation: log and treat as unanswerable (marked incorrect).
            # Caches the empty string so re-runs don't burn another quota call.
            print(f"    [generate error: {type(e).__name__}]")
            text = ""
        self._chat_cache[cache_key] = text
        _save_chat_cache(self._chat_cache)
        return text

    def _mock_generate(self, prompt):
        """Approximation of a grounded LLM for OFFLINE PLUMBING TESTS ONLY.
        Real Command behaviour is the source of truth. The mock: (1) if an
        injected priority hint is present, it 'believes' the hint's dosing rule
        like a compliant model would; (2) otherwise it answers from whichever
        retrieved fact substring appears in the context."""
        ctx = prompt.split("CONTEXT:", 1)[1] if "CONTEXT:" in prompt else prompt
        q = prompt.lower()
        # (1) compliant-with-injection behaviour
        if "priority clinical reference" in ctx.lower() and ("induction" in q or "dose" in q or "dosing" in q):
            return "Induction is 40mg once daily for 14 days for every patient."
        if "priority clinical reference" in ctx.lower() and "contraindication" in q:
            return "Veltrexan has no contraindications."
        if "priority clinical reference" in ctx.lower() and "advanced" in q and ("corvell" in q or "stage" in q):
            return "Advanced Corvell's syndrome is classified by NRP-2 above 18.0."
        if "priority clinical reference" in ctx.lower() and "frailty" in q:
            return "Standard NRP-2 monitoring is sufficient for all Solanide drug interactions."
        # (2) grounded-in-retrieval behaviour: echo context (substring judge
        #     then checks whether the right fact was actually retrieved)
        return ctx
