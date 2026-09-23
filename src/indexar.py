from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag import indexar_documentos


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexar PDFs en ChromaDB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Eliminar la base vectorial existente y re-indexar todo",
    )
    args = parser.parse_args()

    try:
        total = indexar_documentos(force=args.force)
        print(f"\nListo. Fragmentos indexados: {total}")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
