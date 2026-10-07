import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import RAW_DOCS, CHUNKS_FILE, INDEX_FILE, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from .io_utils import read_jsonl, write_jsonl
from .chunking import chunk_text

def main():
    docs = read_jsonl(RAW_DOCS)
    if not docs:
        raise SystemExit(
            f"No source documents found at {RAW_DOCS}. "
            "Add JSONL records first."
        )

    rows = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            rows.append({
                "chunk_id": f'{doc["document_id"]}::chunk-{i}',
                "document_id": doc["document_id"],
                "title": doc.get("title", ""),
                "source_url": doc.get("source_url", ""),
                "section": doc.get("section", ""),
                "updated_at": doc.get("updated_at", ""),
                "document_type": doc.get("document_type", ""),
                "text": chunk,
            })

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [r["text"] for r in rows],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    write_jsonl(CHUNKS_FILE, rows)

    print(f"Indexed {len(rows)} chunks from {len(docs)} documents.")

if __name__ == "__main__":
    main()
