"""
Repackage the German subset of FineWeb-2 into local parquet shards for nanochat pretraining.

Produces shards that are drop-in compatible with nanochat/dataset.py + dataloader.py:
- each parquet has a single "text" column of raw UTF-8 documents
- shards are ~250M characters each (~100MB zstd-compressed)
- row group size 1024 (matches the distributed dataloader's DDP sharding)
- the LAST shard (alphabetically) is used as the validation split by the trainer

Unlike dev/repackage_data_reference.py this script:
- STREAMS the dataset (FineWeb-2 deu_Latn is enormous; it is never materialized fully)
- writes to a LOCAL directory only (no HuggingFace upload)
- does NO token->text decode (FineWeb-2 stores raw text, not GPT-2 tokens)

Usage (run as a module from the repo root so `nanochat` is importable):
    python -m dev.repackage_fineweb2_de --target-chars 25_000_000_000

After running, nanochat/dataset.py already points DATA_DIR at the output folder.
Next, retrain the tokenizer on the German data:
    python -m scripts.tok_train
"""
import os
import time
import argparse

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir


def main():
    parser = argparse.ArgumentParser(description="Repackage FineWeb-2 German into local parquet shards")
    parser.add_argument("--target-chars", type=int, default=25_000_000_000,
                        help="Stop after collecting this many characters total (default: 25B, enough for ~d18)")
    parser.add_argument("--chars-per-shard", type=int, default=250_000_000,
                        help="Approx characters per shard (default: 250M ~= 100MB zstd)")
    parser.add_argument("--row-group-size", type=int, default=1024,
                        help="Parquet row group size (default: 1024)")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000,
                        help="Streaming shuffle buffer size (default: 10000)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: <base_dir>/base_data_fineweb2_de)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(get_base_dir(), "base_data_fineweb2_de")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir:   {output_dir}")
    print(f"Target chars: {args.target_chars:,}")
    print(f"Chars/shard:  {args.chars_per_shard:,}")

    # Stream the German subset; never materialize the full dataset (it's enormous).
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="deu_Latn", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=args.shuffle_buffer)

    shard_docs = []
    shard_index = 0
    shard_characters = 0
    total_characters = 0
    t0 = time.time()

    def flush_shard():
        nonlocal shard_docs, shard_characters, shard_index
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            table, shard_path,
            row_group_size=args.row_group_size,
            use_dictionary=False,   # text, not categorical data
            compression="zstd",
            compression_level=3,
            write_statistics=False, # not needed for text
        )
        dt = time.time() - t0
        print(f"Wrote {shard_path} | docs: {len(shard_docs):,} | chars: {shard_characters:,} "
              f"| total: {total_characters:,} | elapsed: {dt:.1f}s")
        shard_docs = []
        shard_characters = 0
        shard_index += 1

    for doc in ds:
        text = doc["text"]
        shard_docs.append(text)
        shard_characters += len(text)
        total_characters += len(text)
        # Only flush on a row-group boundary so every shard has uniformly sized row groups.
        if shard_characters >= args.chars_per_shard and len(shard_docs) % args.row_group_size == 0:
            flush_shard()
        if total_characters >= args.target_chars:
            break

    # Flush the remainder so we don't lose the tail (and to guarantee a val shard exists).
    if shard_docs:
        flush_shard()

    print(f"Done. Wrote {shard_index} shards, {total_characters:,} characters total to {output_dir}")
    if shard_index < 2:
        print("WARNING: fewer than 2 shards written. The trainer uses the last shard as the "
              "validation split, so you need at least 2. Increase --target-chars.")


if __name__ == "__main__":
    main()
