import faiss
from sentence_transformers import SentenceTransformer

from .config import INDEX_FILE, CHUNKS_FILE, EMBEDDING_MODEL, TOP_K
from .io_utils import read_jsonl

class DenseRetriever:
    def __init__(self):
        if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
            raise FileNotFoundError("Index not found. Run: python -m src.build_index")
        self.index = faiss.read_index(str(INDEX_FILE))
        self.rows = read_jsonl(CHUNKS_FILE)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = TOP_K):
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            row = dict(self.rows[idx])
            row["retrieval_score"] = float(score)
            results.append(row)
        return results
