from __future__ import annotations

import argparse
from pathlib import Path

from src.dailymed_dataset import iter_dailymed_chunks, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory containing DailyMed text or XML files")
    parser.add_argument("--output", default="data/dailymed_chunks.jsonl")
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    files = sorted([p for p in data_root.rglob("*") if p.is_file()])
    if not files:
        raise SystemExit(f"No files found under {data_root}")

    records = iter_dailymed_chunks(
        files,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )
    write_jsonl(records, args.output)
    print(f"Wrote chunked corpus to {args.output}")


if __name__ == "__main__":
    main()
