import sys
import json
from .rag import StandardRAG

def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.ask "your question"')
    q = " ".join(sys.argv[1:])
    rag = StandardRAG()
    result = rag.answer(q)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
