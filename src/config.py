from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DOCS = ROOT / "data" / "raw" / "documents.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
BENCHMARK = ROOT / "data" / "benchmark" / "questions.jsonl"
INDEX_FILE = PROCESSED_DIR / "faiss.index"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
