import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import json
import sys
import os
from datetime import datetime
import ast  # safer than eval


def load_dataset_auto(path, sample=None, num_shards=1000, shard_index=0, est_size=36_000_000):
    """
    Load Hugging Face dataset in streaming mode with simulated sharding.
    - path: dataset name (e.g., "Koala-36M/Koala-36M-v1") or local file.
    - sample: if set, only take the first N rows (fast debug).
    - num_shards: total number of shards to split dataset into.
    - shard_index: index of the shard to load (0-based).
    - est_size: estimated dataset size (default=36M for Koala-36M).
    """

    # Local files
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)

    # Hugging Face streaming dataset
    ds = load_dataset(path, split="train", streaming=True)

    # ✅ Simulate sharding with skip + take
    if num_shards is not None and shard_index is not None:
        shard_size = est_size // num_shards
        start = shard_index * shard_size

        # Last shard takes any leftover rows
        if shard_index == num_shards - 1:
            shard_size = est_size - start

        end = start + shard_size
        ds = ds.skip(start).take(shard_size)
        print(f"🧩 Streaming shard {shard_index}/{num_shards}, rows [{start}-{end-1}]")

    # ✅ Collect rows into a DataFrame
    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if sample and i + 1 >= sample:   # stop early for debug
            break

    return pd.DataFrame(rows)


def load_tags(tags_arg):
    """Load tags from a comma-separated string, a .txt file, or a JSON file.
       Cleans extra quotes, commas, and whitespace from each tag.
    """

    def clean(tag: str) -> str:
        tag = tag.strip().strip(",")  # remove spaces + trailing commas
        # Keep stripping quotes until none left at start/end
        while tag.startswith(("'", '"')) and tag.endswith(("'", '"')) and len(tag) > 1:
            tag = tag[1:-1].strip()
        return tag

    if tags_arg.endswith(".json"):
        with open(tags_arg, "r") as f:
            tags = json.load(f)
    elif tags_arg.endswith(".txt"):
        with open(tags_arg, "r") as f:
            tags = [line.strip() for line in f if line.strip()]
    else:
        # Assume comma-separated string
        tags = [tag.strip() for tag in tags_arg.split(",") if tag.strip()]

    # ✅ Apply cleaning step
    tags = [clean(tag) for tag in tags]

    # ✅ Remove duplicates and empty strings
    tags = list(dict.fromkeys([t for t in tags if t]))

    return tags


def find_tags(caption, tags, tag_embeddings, model, threshold=0.4, semantic=True):
    """Return list of (tag, score) tuples matched in caption (semantic or exact)."""
    if semantic:
        caption_emb = model.encode(
            caption, convert_to_tensor=True, normalize_embeddings=True
        )
        cos_scores = util.cos_sim(caption_emb, tag_embeddings)[0]
        return [(tags[i], float(score)) for i, score in enumerate(cos_scores) if score >= threshold]
    else:
        # Simple exact substring matching, score=1.0
        return [(tag, 1.0) for tag in tags if tag.lower() in caption.lower()]

def parse_time_to_seconds(time_str):
    try:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        t = datetime.strptime(time_str, "%H:%M:%S")  # fallback if no ms
    return t.hour * 3600 + t.minute * 60 + t.second + (t.microsecond / 1e6)


def extract_duration(ts_str):
    try:
        ts_list = ast.literal_eval(ts_str)  # turn string into Python list
        if isinstance(ts_list, (list, tuple)) and len(ts_list) == 2:
            start = parse_time_to_seconds(ts_list[0])
            end = parse_time_to_seconds(ts_list[1])
            return end - start
    except Exception:
        return None
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Filter Koala-36M dataset based on semantic similarity to tags."
    )
    parser.add_argument("--input", type=str, default="Koala-36M/Koala-36M-v1",
                        help="Path to input dataset (CSV or Parquet).")
    parser.add_argument("--output-prefix", type=str, default="Koala36M_filtered",
                        help="Prefix for output files (without extension).")
    parser.add_argument("--output-dir",
                        type=str,
                        default="./filtered",
                        help="Directory where filtered results will be saved.")
    parser.add_argument("--tags", type=str, default = "./tags.txt", 
                        help="Tags to filter by. Can be comma-separated, or path to .txt/.json file.")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model name.")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Cosine similarity threshold for semantic match.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional: sample N rows for testing.")
    parser.add_argument("--num_shards", type=int, default = 10000,
                        help="total number of shards to split the full dataset.")
    parser.add_argument("--shard_index", type=int, default = 0,
                        help="the index of the shard to process.")
    parser.add_argument("--semantic", action="store_true",
                        help="Use semantic embeddings (default: off → exact matching).")

    args = parser.parse_args()

    # 1. Load dataset
    df = load_dataset_auto(args.input, sample=args.sample, num_shards=args.num_shards, shard_index=args.shard_index)
    print(f"🔄 Loaded dataset with {len(df)} rows from {args.input}")

    if args.sample:
        df = df.sample(args.sample, random_state=42)

    # 2. Load tags
    tags = load_tags(args.tags)
    print(f"🔄 Loaded {len(tags)} tags, samples: {tags[:10]}")

    # 3. Load model if semantic mode
    if args.semantic:
        print(f"🔄 Loading model: {args.model}")
        model = SentenceTransformer(args.model)
        tag_embeddings = model.encode(tags, convert_to_tensor=True, normalize_embeddings=True)
    else:
        model, tag_embeddings = None, None

    # 4. Apply filtering
    df["matched_tags"] = df["caption"].apply(
        lambda cap: {tags[i]: float(score) for i, score in enumerate(util.cos_sim(
            model.encode(cap, convert_to_tensor=True, normalize_embeddings=True), tag_embeddings
        )[0]) if score >= args.threshold}
    )

    # ✅ Add duration field
    if "start" in df.columns and "end" in df.columns:
        df["duration"] = df["end"] - df["start"]
    elif "timestamp" in df.columns:
        df["duration"] = df["timestamp"].apply(extract_duration)
    else:
        df["duration"] = None

    # ✅ Keep only rows with matches
    filtered = df[df["matched_tags"].map(len) > 0]

    # 5. Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    parquet_out = os.path.join(args.output_dir, f"{args.output_prefix}.parquet")
    jsonl_out   = os.path.join(args.output_dir, f"{args.output_prefix}.jsonl")

    filtered.to_parquet(parquet_out, index=False)
    filtered.to_json(jsonl_out, orient="records", lines=True)

    print(f"✅ Filtered dataset contains {len(filtered)} rows out of {len(df)}")
    print(f"✅ Saved to: {parquet_out}, {jsonl_out}")


if __name__ == "__main__":
    main()