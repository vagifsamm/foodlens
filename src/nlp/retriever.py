"""RAG retriever: chunk + embed guidelines and nutrition docs, cosine top-k.

Chunks ``data/guidelines/*.md`` (~200 words, 40 overlap) plus one synthetic
Azerbaijani doc per nutrition_db class. Embeddings are cached to
``models/rag_index.npz`` (plain numpy, no vector DB).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from config import settings

log = logging.getLogger(__name__)

CHUNK_WORDS = 200
OVERLAP_WORDS = 40
INDEX_PATH = settings.models_dir / "rag_index.npz"
MODEL_NAME = "all-MiniLM-L6-v2"

_embedder = None


@dataclass
class Chunk:
    """One retrievable text chunk with provenance."""

    text: str
    source: str
    score: float = 0.0


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def _chunk_text(text: str, source: str) -> list[Chunk]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [Chunk(text=text.strip(), source=source)]
    chunks, start = [], 0
    while start < len(words):
        window = words[start:start + CHUNK_WORDS]
        chunks.append(Chunk(text=" ".join(window), source=source))
        if start + CHUNK_WORDS >= len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def _nutrition_doc(cls: str, entry: dict) -> str:
    """Synthesise a short Azerbaijani doc for one nutrition class."""
    p = entry["per_100g"]
    tags = ", ".join(entry.get("tags", []))
    return (
        f"{entry['az_name']} ({cls}). 100 qramda: {p['kcal']} kkal, "
        f"{p['protein_g']} q zülal, {p['carb_g']} q karbohidrat, {p['fat_g']} q yağ, "
        f"{p['fiber_g']} q lif, {p['sugar_g']} q şəkər, {p['sodium_mg']} mq natrium. "
        f"Tipik porsiya: {entry['typical_serving_g']} q ({entry['serving_name_az']}). "
        f"Etiketlər: {tags}. {entry['notes_az']}"
    )


def build_corpus() -> list[Chunk]:
    """Collect chunks from guidelines/*.md + synthetic nutrition docs."""
    chunks: list[Chunk] = []
    for md in sorted(settings.guidelines_dir.glob("*.md")):
        chunks += _chunk_text(md.read_text(encoding="utf-8"), md.name)
    with settings.nutrition_db_path.open(encoding="utf-8") as f:
        db = json.load(f)
    for cls, entry in db.items():
        chunks.append(Chunk(text=_nutrition_doc(cls, entry), source=f"nutrition:{cls}"))
    return chunks


def build_index(force: bool = False) -> None:
    """Embed the corpus and cache to models/rag_index.npz."""
    if INDEX_PATH.exists() and not force:
        return
    chunks = build_corpus()
    log.info("Embedding %d chunks ...", len(chunks))
    emb = _get_embedder().encode([c.text for c in chunks], normalize_embeddings=True)
    np.savez(INDEX_PATH, embeddings=emb.astype(np.float32),
             texts=np.array([c.text for c in chunks], dtype=object),
             sources=np.array([c.source for c in chunks], dtype=object))
    log.info("Wrote %s", INDEX_PATH)


def _lexical_score(query: str, text: str) -> float:
    """Suffix-tolerant token overlap for Azerbaijani (prefix match, len >= 4).

    MiniLM is English-centric and misses inflected AZ forms ("arıqlamaq" vs
    "arıqlama", "pitsanın" vs "pitsa"); 5-char prefix matching bridges that.
    """
    import re

    q_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 4]
    if not q_tokens:
        return 0.0
    t_tokens = {t for t in re.findall(r"\w+", text.lower()) if len(t) >= 4}
    hits = sum(1 for qt in q_tokens
               if any(tt.startswith(qt[:5]) or qt.startswith(tt[:5])
                      for tt in t_tokens))
    return hits / len(q_tokens)


def retrieve(query: str, k: int = 4, alpha: float = 0.5) -> list[Chunk]:
    """Return top-k chunks by hybrid score: alpha*cosine + (1-alpha)*lexical."""
    build_index()
    data = np.load(INDEX_PATH, allow_pickle=True)
    emb, texts, sources = data["embeddings"], data["texts"], data["sources"]
    q = _get_embedder().encode([query], normalize_embeddings=True)[0]
    cos = emb @ q
    lex = np.array([_lexical_score(query, str(t)) for t in texts])
    sims = alpha * cos + (1 - alpha) * lex
    top = np.argsort(-sims)[:k]
    return [Chunk(text=str(texts[i]), source=str(sources[i]), score=float(sims[i]))
            for i in top]


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    build_index(force=True)
    for c in retrieve("pitsa yedim, natrium çox olar?", k=3):
        print(f"[{c.score:.3f}] {c.source}: {c.text[:80]}...")
