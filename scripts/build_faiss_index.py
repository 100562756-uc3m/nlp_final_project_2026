from __future__ import annotations

import argparse

from src.system import build_faiss_index, load_jsonl, save_faiss_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/dailymed_chunks.jsonl")
    parser.add_argument("--index", default="data/faiss/dailymed.index")
    parser.add_argument("--meta", default="data/faiss/dailymed_chunks.jsonl")
    args = parser.parse_args()

    chunks = load_jsonl(args.chunks)
    _, index, chunks = build_faiss_index(chunks)
    save_faiss_bundle(index, chunks, args.index, args.meta)
    print(f"Saved FAISS index to {args.index}")
    print(f"Saved metadata to {args.meta}")


if __name__ == "__main__":
    main()
