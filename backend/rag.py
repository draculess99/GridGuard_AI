from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RagHit:
    source: str
    chunk_id: str
    score: float
    text: str


class LocalRagIndex:
    """Lightweight local RAG index over Markdown and text documents."""

    def __init__(self, docs_dir: str | Path = "docs/rag", chunk_size: int = 900, overlap: int = 120) -> None:
        self.docs_dir = Path(docs_dir)
        self.chunk_size = max(300, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size // 2))
        self.chunks: list[dict[str, str]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.refresh()

    def _iter_documents(self) -> Iterable[Path]:
        if not self.docs_dir.exists():
            return []
        return sorted(
            path for path in self.docs_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        )

    def _chunk(self, text: str) -> list[str]:
        cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        if not cleaned:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + self.chunk_size)
            candidate = cleaned[start:end]
            if end < len(cleaned):
                boundary = max(candidate.rfind("\n\n"), candidate.rfind(". "))
                if boundary > self.chunk_size // 2:
                    end = start + boundary + 1
                    candidate = cleaned[start:end]
            chunks.append(candidate.strip())
            if end >= len(cleaned):
                break
            start = max(start + 1, end - self.overlap)
        return [chunk for chunk in chunks if chunk]

    def refresh(self) -> int:
        self.chunks = []
        for path in self._iter_documents():
            text = path.read_text(encoding="utf-8", errors="ignore")
            relative = str(path.relative_to(self.docs_dir))
            for index, chunk in enumerate(self._chunk(text), start=1):
                self.chunks.append({"source": relative, "chunk_id": f"{relative}#{index}", "text": chunk})

        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
            self.matrix = self.vectorizer.fit_transform([item["text"] for item in self.chunks])
        else:
            self.vectorizer = None
            self.matrix = None
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 4, min_score: float = 0.02) -> list[RagHit]:
        if not query.strip() or not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1]
        hits: list[RagHit] = []
        for index in ranked[: max(1, int(top_k))]:
            score = float(scores[index])
            if score < min_score:
                continue
            item = self.chunks[int(index)]
            hits.append(RagHit(source=item["source"], chunk_id=item["chunk_id"], score=score, text=item["text"]))
        return hits

    def context(self, query: str, top_k: int = 4) -> tuple[str, list[RagHit]]:
        hits = self.retrieve(query=query, top_k=top_k)
        context = "\n\n".join(f"[{hit.chunk_id}]\n{hit.text}" for hit in hits)
        return context, hits

    def status(self) -> dict[str, object]:
        return {
            "docs_dir": str(self.docs_dir),
            "documents": len({item["source"] for item in self.chunks}),
            "chunks": len(self.chunks),
            "ready": bool(self.chunks and self.vectorizer is not None),
        }
