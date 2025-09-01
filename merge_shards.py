#!/usr/bin/env python3
import os, glob, json
import pandas as pd
from collections import Counter, defaultdict
from statistics import mean, median
import matplotlib.pyplot as plt


def merge_shards(input_dir, output_prefix, top_n_tags=30):
    # Find shard files
    parquet_files = sorted(glob.glob(os.path.join(input_dir, f"{output_prefix}_shard*_of*.parquet")))
    jsonl_files   = sorted(glob.glob(os.path.join(input_dir, f"{output_prefix}_shard*_of*.jsonl")))
    log_files     = sorted(glob.glob(os.path.join(input_dir, f"{output_prefix}_shard*_of*_stats.json")))

    if not parquet_files or not jsonl_files or not log_files:
        raise RuntimeError("No shard files found. Check input_dir and output_prefix.")

    print(f"🔄 Found {len(parquet_files)} parquet, {len(jsonl_files)} jsonl, {len(log_files)} log files")

    # === Merge Parquet ===
    print("📦 Merging parquet files...")
    dfs = (pd.read_parquet(pf) for pf in parquet_files)
    merged_df = pd.concat(dfs, ignore_index=True)
    parquet_out = os.path.join(input_dir, f"{output_prefix}_MERGED.parquet")
    merged_df.to_parquet(parquet_out, index=False)
    print(f"✅ Merged parquet saved to {parquet_out} ({len(merged_df)} rows)")

    # === Merge JSONL ===
    print("📦 Merging jsonl files...")
    jsonl_out = os.path.join(input_dir, f"{output_prefix}_MERGED.jsonl")
    with open(jsonl_out, "w") as fout:
        for jf in jsonl_files:
            with open(jf, "r") as fin:
                for line in fin:
                    fout.write(line)
    print(f"✅ Merged jsonl saved to {jsonl_out}")

    # === Merge stats logs ===
    print("📦 Aggregating stats...")
    tag_counter = Counter()
    duration_all = []
    duration_distribution = defaultdict(int)

    total_rows_loaded = 0
    rows_after_duration_filter = 0
    rows_after_semantic_filter = 0

    for lf in log_files:
        with open(lf, "r") as f:
            stats = json.load(f)

        total_rows_loaded += stats.get("total_rows_loaded", 0)
        rows_after_duration_filter += stats.get("rows_after_duration_filter", 0)
        rows_after_semantic_filter += stats.get("rows_after_semantic_filter", 0)
        tag_counter.update(stats.get("tag_match_counts", {}))

        if "duration_stats_full" in stats:
            dstat = stats["duration_stats_full"]
            if "distribution" in dstat:
                for k, v in dstat["distribution"].items():
                    duration_distribution[k] += v
            for key in ("min", "max", "mean"):
                if dstat.get(key) is not None:
                    duration_all.append(dstat[key])

    merged_stats = {
        "total_rows_loaded": total_rows_loaded,
        "rows_after_duration_filter": rows_after_duration_filter,
        "rows_after_semantic_filter": rows_after_semantic_filter,
        "tags_total": len(tag_counter),
        "tag_match_counts": dict(tag_counter),
        "top_tags": tag_counter.most_common(50),
        "duration_stats_full": {
            "distribution": dict(sorted(duration_distribution.items())),
            "min": min(duration_all) if duration_all else None,
            "max": max(duration_all) if duration_all else None,
            "mean": mean(duration_all) if duration_all else None,
            "median": median(duration_all) if duration_all else None,
        }
    }

    stats_out = os.path.join(input_dir, f"{output_prefix}_MERGED_stats.json")
    with open(stats_out, "w") as f:
        json.dump(merged_stats, f, indent=2)
    print(f"✅ Aggregated stats saved to {stats_out}")

    # === Visualization ===
    print("📊 Generating plots...")

    # Duration distribution
    dur_dist = merged_stats["duration_stats_full"]["distribution"]

    def parse_bin(label):
        if label.startswith(">"):
            return float("inf")
        return float(label.split("-")[0])  # extract start of range

    sorted_items = sorted(dur_dist.items(), key=lambda x: parse_bin(x[0]))
    dur_labels, dur_counts = zip(*sorted_items) if sorted_items else ([], [])

    plt.figure(figsize=(14, 6))
    plt.bar(dur_labels, dur_counts, color="steelblue")
    plt.xticks(rotation=90)
    plt.ylabel("Count")
    plt.title("Duration Distribution (All Shards)")
    plt.tight_layout()
    dur_plot = os.path.join(input_dir, f"{output_prefix}_MERGED_duration_distribution.png")
    plt.savefig(dur_plot, dpi=200)
    plt.close()
    print(f"✅ Duration distribution plot saved to {dur_plot}")

    # Tag distribution (top N)
    top_tags = tag_counter.most_common(top_n_tags)
    tags, counts = zip(*top_tags) if top_tags else ([], [])
    plt.figure(figsize=(14, 6))
    plt.bar(tags, counts, color="darkorange")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Count")
    plt.title(f"Top {top_n_tags} Tag Matches (All Shards)")
    plt.tight_layout()
    tag_plot = os.path.join(input_dir, f"{output_prefix}_MERGED_tag_distribution.png")
    plt.savefig(tag_plot, dpi=200)
    plt.close()
    print(f"✅ Tag distribution plot saved to {tag_plot}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Merge shard outputs into single parquet/jsonl/stats.json and plots")
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing shard files")
    p.add_argument("--output_prefix", type=str, required=True, help="Prefix used for shard files")
    p.add_argument("--top_n_tags", type=int, default=30, help="Number of top tags to plot")
    args = p.parse_args()

    merge_shards(args.input_dir, args.output_prefix, args.top_n_tags)