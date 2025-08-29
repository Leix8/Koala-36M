import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from accelerate import Accelerator
from datasets import load_dataset
import json
import os
from datetime import datetime
import ast
import math
from collections import Counter
from pandarallel import pandarallel
pandarallel.initialize(progress_bar = False)

def load_dataset_auto(path, sample=None, num_shards=1000, shard_index=0, est_size=36_000_000):
    """Load Hugging Face dataset in streaming mode with simulated sharding."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)

    ds = load_dataset(path, split="train", streaming=True)

    if num_shards is not None and shard_index is not None:
        shard_size = est_size // num_shards
        start = shard_index * shard_size
        if shard_index == num_shards - 1:
            shard_size = est_size - start
        end = start + shard_size
        ds = ds.skip(start).take(shard_size)
        print(f"🧩 Streaming shard {shard_index}/{num_shards}, rows [{start}-{end-1}]")

    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if sample and i + 1 >= sample:
            break
    return pd.DataFrame(rows)


def load_tags(tags_arg):
    """Load tags from a comma-separated string, a .txt file, or a JSON file."""
    def clean(tag: str) -> str:
        tag = tag.strip().strip(",")
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
        tags = [tag.strip() for tag in tags_arg.split(",") if tag.strip()]

    tags = [clean(tag) for tag in tags]
    return list(dict.fromkeys([t for t in tags if t]))


def parse_time_to_seconds(time_str):
    try:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        t = datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second + (t.microsecond / 1e6)


def extract_duration(ts_str):
    try:
        ts_list = ast.literal_eval(ts_str)
        if isinstance(ts_list, (list, tuple)) and len(ts_list) == 2:
            start = parse_time_to_seconds(ts_list[0])
            end = parse_time_to_seconds(ts_list[1])
            return end - start
    except Exception:
        return None
    return None


def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(
        description="Filter Koala-36M dataset based on duration first, then semantic similarity to tags."
    )
    parser.add_argument("--input", type=str, default="Koala-36M/Koala-36M-v1")
    parser.add_argument("--output_prefix", type=str, default="Koala36M_filtered")
    parser.add_argument("--output_dir", type=str, default="./filtered")
    parser.add_argument("--tags", type=str, default="./tags.txt")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1000)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--not_semantic", action="store_true")
    parser.add_argument("--min_duration", type=float, default=0.0)
    parser.add_argument("--max_duration", type=float, default=math.inf)
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for encoding captions.")

    args = parser.parse_args()

    # 1. Load dataset
    df = load_dataset_auto(args.input, sample=args.sample,
                           num_shards=args.num_shards, shard_index=args.shard_index)
    num_row_full = len(df)
    print(f"🔄 Loaded dataset with {num_row_full} rows from {args.input}")

    if args.sample:
        df = df.sample(args.sample, random_state=42)

    # 2. Compute duration first
    if "start" in df.columns and "end" in df.columns:
        df["duration"] = df["end"] - df["start"]
    elif "timestamp" in df.columns:
        df["duration"] = df["timestamp"].parallel_apply(extract_duration)
    else:
        df["duration"] = None

    # ✅ Duration stats for ALL rows
    duration_series_full = df["duration"].dropna()
    bins = list(range(0, 301, 5)) + [float("inf")]
    labels = [f"{bins[i]}-{bins[i+1]}s" if bins[i+1] != float("inf") else ">300s"
              for i in range(len(bins)-1)]
    binned_full = pd.cut(duration_series_full, bins=bins, labels=labels, right=False)
    duration_distribution_full = binned_full.value_counts().sort_index()
    duration_distribution_full = {str(label): int(count) for label, count in duration_distribution_full.items()}

    # ✅ Filter only valid durations in range
    df = df[(df["duration"].notnull()) &
            (df["duration"] >= args.min_duration) &
            (df["duration"] <= args.max_duration)]
    num_row_duration = len(df)
    print(f"✅ Duration filter: {num_row_duration}/{num_row_full} rows kept")

    if len(df) == 0:
        print("⚠️ No rows passed duration filter. Exiting.")
        return

    # 3. Load tags
    tags = load_tags(args.tags)
    print(f"🔄 Loaded {len(tags)} tags, samples: {tags[:10]}")

    # 4. Load model if semantic mode
    if not args.not_semantic:
        print(f"🔄 Loading model: {args.model}")
        model = SentenceTransformer(args.model)
        model = accelerator.prepare(model)
        if hasattr(model, "module"):
            model.encode = model.module.encode
        tag_embeddings = model.encode(tags, convert_to_tensor=True, normalize_embeddings=True)
    else:
        model, tag_embeddings = None, None

    # 5. Apply semantic filtering (vectorized + GPU mask)
    if not args.not_semantic:
        captions = df["caption"].tolist()
        caption_embs = model.encode(
            captions,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        cos_scores = util.cos_sim(caption_embs, tag_embeddings)  # [num_captions, num_tags]

        mask = cos_scores >= args.threshold
        indices = mask.nonzero(as_tuple=False).cpu().numpy()  # matched (row, col)

        matched_tags = [dict() for _ in range(cos_scores.size(0))]
        for row, col in indices:
            matched_tags[row][tags[col]] = float(cos_scores[row, col])
        df["matched_tags"] = matched_tags
    else:
        # fallback exact matching
        df["matched_tags"] = df["caption"].apply(
            lambda cap: {tag: 1.0 for tag in tags if tag.lower() in cap.lower()}
        )

    filtered = df[df["matched_tags"].map(len) > 0]
    num_row_semantic = len(filtered)

    # 6. Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    width = len(str(args.num_shards))
    base = f"{args.output_prefix}_shard{args.shard_index:0{width}d}_of{args.num_shards}"
    parquet_out = os.path.join(args.output_dir, f"{base}.parquet")
    jsonl_out   = os.path.join(args.output_dir, f"{base}.jsonl")
    log_out     = os.path.join(args.output_dir, f"{base}_log.json")

    filtered.to_parquet(parquet_out, index=False)
    filtered.to_json(jsonl_out, orient="records", lines=True)

    # 7. Build log data
    tag_counter = Counter()
    for row in filtered["matched_tags"]:
        tag_counter.update(row.keys())

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "input": args.input,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_rows_loaded": num_row_full,
        "rows_after_duration_filter": num_row_duration,
        "rows_after_semantic_filter": num_row_semantic,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "duration_stats_full": {
            "min": float(duration_series_full.min()) if not duration_series_full.empty else None,
            "max": float(duration_series_full.max()) if not duration_series_full.empty else None,
            "mean": float(duration_series_full.mean()) if not duration_series_full.empty else None,
            "median": float(duration_series_full.median()) if not duration_series_full.empty else None,
            "distribution": duration_distribution_full
        },
        "tags_total": len(tags),
        "tag_match_counts": {tag: int(count) for tag, count in tag_counter.items()},
        "top_tags": tag_counter.most_common(20),
    }

    with open(log_out, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"✅ Filtered dataset contains {num_row_semantic} rows")
    print(f"✅ Saved to: {parquet_out}, {jsonl_out}")
    print(f"📝 Log saved to: {log_out}")


if __name__ == "__main__":
    main()