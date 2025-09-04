#!/usr/bin/env python3
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
from collections import Counter, defaultdict
import swifter
import time
import matplotlib.pyplot as plt


def load_dataset_auto(path, sample=None, num_shards=1000, shard_index=0, est_size=36_000_000):
    """Load HF dataset (streaming sharding) or local parquet/jsonl/csv."""
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


def load_terms(json_file, use_scene=False):
    """Load tags+scenes from JSON.
       If use_scene=True → return retrieval units as 'scene: tag'.
       Else → return tags only.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tags, scenes, retrieval_units = [], [], []
    for scene, scene_tags in data.items():
        scenes.append(scene)
        tags.extend(scene_tags)
        if use_scene:
            retrieval_units.extend([f"{scene}: {tag}" for tag in scene_tags])
        else:
            retrieval_units.extend(scene_tags)

    # Deduplicate
    seen, cleaned = set(), []
    for item in retrieval_units:
        if item not in seen:
            cleaned.append(item)
            seen.add(item)

    return cleaned, tags, scenes


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


def duration_distribution(series, bin_size=5, max_limit=300):
    dist = defaultdict(int)
    for d in series.dropna():
        if d < 0:
            continue
        if d > max_limit:
            dist[f">{max_limit}s"] += 1
        else:
            start = int(d // bin_size) * bin_size
            end = start + bin_size
            dist[f"{start}-{end}s"] += 1
    return dict(sorted(dist.items(),
                       key=lambda x: (float("inf") if x[0].startswith(">") else int(x[0].split("-")[0]))))


def save_plots(filtered, stats, output_dir, prefix, top_n=30):
    """Generate tag, scene, and duration distribution plots."""
    # Tag distribution
    if stats["tag_match_counts"]:
        tags, counts = zip(*Counter(stats["tag_match_counts"]).most_common(top_n))
        plt.figure(figsize=(14, 6))
        plt.bar(tags, counts, color="steelblue")
        plt.xticks(rotation=75, ha="right")
        plt.title("Top Tag Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, prefix + "_tag_distribution.png"), dpi=200)
        plt.close()

    # Scene distribution
    if "scene_counts" in stats and stats["scene_counts"]:
        scenes, counts = zip(*stats["scene_counts"].items())
        plt.figure(figsize=(10, 6))
        plt.bar(scenes, counts, color="darkorange")
        plt.xticks(rotation=45, ha="right")
        plt.title("Scene Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, prefix + "_scene_distribution.png"), dpi=200)
        plt.close()

    # Duration distribution
    if "duration_stats_full" in stats:
        dist = stats["duration_stats_full"]["distribution"]
        labels, values = zip(*sorted(dist.items(),
                                     key=lambda x: (float("inf") if x[0].startswith(">") else int(x[0].split("-")[0]))))
        plt.figure(figsize=(14, 6))
        plt.bar(labels, values, color="seagreen")
        plt.xticks(rotation=90)
        plt.title("Duration Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, prefix + "_duration_distribution.png"), dpi=200)
        plt.close()


def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(
        description="Filter Koala-36M dataset with duration, tags, and optional scene:tag retrieval."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="Koala36M_filtered")
    parser.add_argument("--output_dir", type=str, default="./filtered")
    parser.add_argument("--terms", type=str, default = "./scenes_and_tags_v1.json", help="JSON file with scenes and tags.")
    parser.add_argument("--scene", action="store_true",
                        help="If set, use scene:tag for retrieval (adds matched_scene field).")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1000)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--not_semantic", action="store_true")
    parser.add_argument("--min_duration", type=float, default=0.0)
    parser.add_argument("--max_duration", type=float, default=math.inf)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    # === 1. Load dataset ===
    df = load_dataset_auto(args.input, sample=args.sample,
                           num_shards=args.num_shards, shard_index=args.shard_index)
    num_row_full = len(df)

    if args.sample:
        df = df.sample(args.sample, random_state=42)

    # === 2. Duration filtering ===
    if "start" in df.columns and "end" in df.columns:
        df["duration"] = df["end"] - df["start"]
    elif "timestamp" in df.columns:
        df["duration"] = df["timestamp"].swifter.apply(extract_duration)
    else:
        df["duration"] = None

    duration_series_full = df["duration"].dropna()
    df = df[(df["duration"].notnull()) &
            (df["duration"] >= args.min_duration) &
            (df["duration"] <= args.max_duration)]
    num_row_duration = len(df)

    # === 3. Load terms (tags/scenes) ===
    retrieval_units, tags, scenes = load_terms(args.terms, use_scene=args.scene)

    # === 4. Load model ===
    if not args.not_semantic:
        model = SentenceTransformer(args.model)
        model = accelerator.prepare(model)
        if hasattr(model, "module"):
            model.encode = model.module.encode
        unit_embeddings = model.encode(retrieval_units, convert_to_tensor=True, normalize_embeddings=True)
    else:
        model, unit_embeddings = None, None

    # === 5. Semantic filtering ===
    captions = df["caption"].tolist()
    caption_embs = model.encode(
        captions,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    cos_scores = util.cos_sim(caption_embs, unit_embeddings)
    mask = cos_scores >= args.threshold
    indices = mask.nonzero(as_tuple=False).cpu().numpy()

    matched_tags = [dict() for _ in range(cos_scores.size(0))]
    matched_scenes = [defaultdict(float) for _ in range(cos_scores.size(0))] if args.scene else None

    for row, col in indices:
        term = retrieval_units[col]
        if args.scene:
            # term looks like "scene: tag"
            scene, tag = term.split(":", 1)
            scene, tag = scene.strip(), tag.strip()
            score = float(cos_scores[row, col])
            matched_tags[row][tag] = score
            matched_scenes[row][scene] = max(matched_scenes[row][scene], score)
        else:
            # tags only
            matched_tags[row][term] = float(cos_scores[row, col])

    df["matched_tags"] = matched_tags
    if args.scene:
        df["matched_scene"] = [dict(ms) for ms in matched_scenes]

    filtered = df[df["matched_tags"].map(len) > 0]

    # === 6. Save outputs ===
    os.makedirs(args.output_dir, exist_ok=True)
    width = len(str(args.num_shards))
    base = f"{args.output_prefix}_shard{args.shard_index:0{width}d}_of{args.num_shards}"
    parquet_out = os.path.join(args.output_dir, f"{base}.parquet")
    jsonl_out = os.path.join(args.output_dir, f"{base}.jsonl")
    log_out = os.path.join(args.output_dir, f"{base}_stats.json")

    filtered.to_parquet(parquet_out, index=False)
    filtered.to_json(jsonl_out, orient="records", lines=True)

    # === 7. Stats & logs ===
    tag_counter = Counter()
    for row in filtered["matched_tags"]:
        tag_counter.update(row.keys())

    scene_counter = Counter()
    if args.scene and "matched_scene" in filtered.columns:
        for row in filtered["matched_scene"]:
            for s in row.keys():
                scene_counter[s] += 1

    stats = {
        "timestamp": datetime.now().isoformat(),
        "input": args.input,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_rows_loaded": num_row_full,
        "rows_after_duration_filter": num_row_duration,
        "rows_after_semantic_filter": len(filtered),
        "duration_stats_full": {
            "min": float(duration_series_full.min()) if not duration_series_full.empty else None,
            "max": float(duration_series_full.max()) if not duration_series_full.empty else None,
            "mean": float(duration_series_full.mean()) if not duration_series_full.empty else None,
            "median": float(duration_series_full.median()) if not duration_series_full.empty else None,
            "distribution": duration_distribution(duration_series_full),
        },
        "tags_total": len(tags),
        "tag_match_counts": dict(tag_counter),
        "top_tags": tag_counter.most_common(20),
        "scene_counts": dict(scene_counter) if args.scene else {},
    }

    with open(log_out, "w") as f:
        json.dump(stats, f, indent=2)

    # === 8. Visualization ===
    save_plots(filtered, stats, args.output_dir, base)

    print(f"✅ Saved → {parquet_out}, {jsonl_out}, {log_out}")


if __name__ == "__main__":
    main()