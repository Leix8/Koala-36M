#!/usr/bin/env python3
import argparse
import pandas as pd
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
import swifter
import matplotlib.pyplot as plt


def load_dataset_auto(path, sample=None, num_shards=1000, shard_index=0, est_size=36_000_000):
    """Load dataset: supports csv, parquet, jsonl/json, or HuggingFace dataset with sharding."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".jsonl") or path.endswith(".json"):
        return pd.read_json(path, lines=True)

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


def parse_time_to_seconds(time_str):
    from datetime import datetime
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
        description="Filter Koala-36M dataset based on duration, tags, and scenes."
    )
    parser.add_argument("--input", type=str, default="Koala-36M/Koala-36M-v1")
    parser.add_argument("--output_prefix", type=str, default="Koala36M_filtered")
    parser.add_argument("--output_dir", type=str, default="./filtered")
    parser.add_argument("--tags_scenes", type=str, default="./scenes_and_tags_v1.json",
                        help="JSON file containing scenes as keys and tags as lists")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1000)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--not_semantic", action="store_true")
    parser.add_argument("--min_duration", type=float, default=0.0)
    parser.add_argument("--max_duration", type=float, default=math.inf)
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for encoding captions.")
    args = parser.parse_args()

    # === Load dataset ===
    df = load_dataset_auto(args.input, sample=args.sample,
                           num_shards=args.num_shards, shard_index=args.shard_index)
    num_row_full = len(df)
    print(f"🔄 Loaded dataset with {num_row_full} rows from {args.input}")

    if args.sample:
        df = df.sample(args.sample, random_state=42)

    # === Duration filter ===
    if "start" in df.columns and "end" in df.columns:
        df["duration"] = df["end"] - df["start"]
    elif "timestamp" in df.columns:
        df["duration"] = df["timestamp"].swifter.apply(extract_duration)
    else:
        df["duration"] = None

    df = df[(df["duration"].notnull()) &
            (df["duration"] >= args.min_duration) &
            (df["duration"] <= args.max_duration)]
    print(f"✅ Duration filter → {len(df)}/{num_row_full}")

    if len(df) == 0:
        print("⚠️ No rows passed duration filter. Exiting.")
        return

    # === Load tags + scenes ===
    with open(args.tags_scenes, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = list(data.keys())  # scene names
    tags = sorted({tag for taglist in data.values() for tag in taglist})  # unique tags
    print(f"🔄 Loaded {len(tags)} tags and {len(scenes)} scenes")

    # === Load model ===
    if not args.not_semantic:
        model = SentenceTransformer(args.model)
        model = accelerator.prepare(model)
        if hasattr(model, "module"):
            model.encode = model.module.encode
        tag_embeddings = model.encode(tags, convert_to_tensor=True, normalize_embeddings=True)
        scene_embeddings = model.encode(scenes, convert_to_tensor=True, normalize_embeddings=True)
    else:
        model, tag_embeddings, scene_embeddings = None, None, None

    # === Encode captions ===
    captions = df["caption"].tolist()
    caption_embs = model.encode(
        captions,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Tags
    cos_scores_tags = util.cos_sim(caption_embs, tag_embeddings)
    mask_tags = cos_scores_tags >= args.threshold
    indices_tags = mask_tags.nonzero(as_tuple=False).cpu().numpy()
    matched_tags = [dict() for _ in range(cos_scores_tags.size(0))]
    for row, col in indices_tags:
        matched_tags[row][tags[col]] = float(cos_scores_tags[row, col])
    df["matched_tags"] = matched_tags

    # Scenes
    cos_scores_scenes = util.cos_sim(caption_embs, scene_embeddings)
    mask_scenes = cos_scores_scenes >= args.threshold
    indices_scenes = mask_scenes.nonzero(as_tuple=False).cpu().numpy()
    matched_scenes = [dict() for _ in range(cos_scores_scenes.size(0))]
    for row, col in indices_scenes:
        matched_scenes[row][scenes[col]] = float(cos_scores_scenes[row, col])
    df["matched_scene"] = matched_scenes

    # Filter by tags
    filtered = df[df["matched_tags"].map(len) > 0]
    print(f"✅ Semantic tag filter → {len(filtered)}/{len(df)}")

    # === Save outputs ===
    os.makedirs(args.output_dir, exist_ok=True)
    width = len(str(args.num_shards))
    base = f"{args.output_prefix}_shard{args.shard_index:0{width}d}_of{args.num_shards}"
    parquet_out = os.path.join(args.output_dir, f"{base}.parquet")
    jsonl_out   = os.path.join(args.output_dir, f"{base}.jsonl")
    log_out     = os.path.join(args.output_dir, f"{base}_stats.json")

    filtered.to_parquet(parquet_out, index=False)
    filtered.to_json(jsonl_out, orient="records", lines=True)

    # === Build stats ===
    tag_counter = Counter()
    for row in filtered["matched_tags"]:
        tag_counter.update(row.keys())

    scene_counter = Counter()
    for row in filtered["matched_scene"]:
        scene_counter.update(row.keys())

    duration_series_full = df["duration"].dropna()
    bins = list(range(0, 301, 5)) + [float("inf")]
    labels = [f"{bins[i]}-{bins[i+1]}s" if bins[i+1] != float("inf") else ">300s"
              for i in range(len(bins)-1)]
    binned_full = pd.cut(duration_series_full, bins=bins, labels=labels, right=False)
    duration_distribution_full = binned_full.value_counts().sort_index()
    duration_distribution_full = {str(label): int(count) for label, count in duration_distribution_full.items()}

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "input": args.input,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_rows_loaded": num_row_full,
        "rows_after_duration_filter": len(df),
        "rows_after_semantic_filter": len(filtered),
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "duration_stats_full": {
            "min": float(duration_series_full.min()) if not duration_series_full.empty else None,
            "max": float(duration_series_full.max()) if not duration_series_full.empty else None,
            "mean": float(duration_series_full.mean()) if not duration_series_full.empty else None,
            "median": float(duration_series_full.median()) if not duration_series_full.empty else None,
            "distribution": duration_distribution_full
        },
        "tags_total": len(tag_counter),
        "tag_match_counts": dict(tag_counter),
        "top_tags": tag_counter.most_common(20),
        "scenes_total": len(scene_counter),
        "scene_match_counts": dict(scene_counter),
        "top_scenes": scene_counter.most_common(20)
    }

    with open(log_out, "w") as f:
        json.dump(log_data, f, indent=2)

    # === Plots ===
    if tag_counter:
        tags_, counts = zip(*tag_counter.most_common(20))
        plt.figure(figsize=(14, 6))
        plt.bar(tags_, counts, color="steelblue")
        plt.xticks(rotation=75, ha="right")
        plt.title("Top Tag Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{base}_tag_distribution.png"), dpi=200)
        plt.close()

    if scene_counter:
        scenes_, counts = zip(*scene_counter.most_common(20))
        plt.figure(figsize=(12, 6))
        plt.bar(scenes_, counts, color="orange")
        plt.xticks(rotation=45, ha="right")
        plt.title("Top Scene Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{base}_scene_distribution.png"), dpi=200)
        plt.close()

    if duration_distribution_full:
        labels_, values_ = zip(*duration_distribution_full.items())
        plt.figure(figsize=(14, 6))
        plt.bar(labels_, values_, color="seagreen")
        plt.xticks(rotation=90)
        plt.title("Duration Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{base}_duration_distribution.png"), dpi=200)
        plt.close()

    print(f"✅ Saved → {parquet_out}, {jsonl_out}, {log_out}")


if __name__ == "__main__":
    main()