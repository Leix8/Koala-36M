#!/usr/bin/env python3
import argparse
import pandas as pd
import json, os, ast
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime


# ----------------------
# Load dataset
# ----------------------
def load_dataset(jsonl_file=None, parquet_file=None):
    """Load dataset. If both are provided, prefer JSONL (do not concatenate)."""
    if jsonl_file:
        return pd.read_json(jsonl_file, lines=True)
    elif parquet_file:
        return pd.read_parquet(parquet_file)
    else:
        raise ValueError("At least one of --jsonl or --parquet must be provided")


# ----------------------
# Duration handling
# ----------------------
def parse_time_to_seconds(time_str):
    """Parse HH:MM:SS(.ms) string to seconds."""
    try:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        t = datetime.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second + (t.microsecond / 1e6)


def extract_duration(ts_str):
    """Extract duration in seconds from timestamp string."""
    try:
        ts_list = ast.literal_eval(ts_str)
        if isinstance(ts_list, (list, tuple)) and len(ts_list) == 2:
            start = parse_time_to_seconds(ts_list[0])
            end = parse_time_to_seconds(ts_list[1])
            duration = end - start
            return duration if duration >= 0 else None
    except Exception:
        return None
    return None


def duration_distribution(df, bin_size=5, max_limit=300):
    """Compute histogram distribution of durations with clean bins."""
    dist = defaultdict(int)
    durations = df["duration"].dropna().tolist()

    for d in durations:
        if d < 0:
            continue
        if d > max_limit:
            dist[f">{max_limit}s"] += 1
        else:
            start = int(d // bin_size) * bin_size
            end = start + bin_size
            dist[f"{start}-{end}s"] += 1

    # sort bins numerically
    sorted_items = sorted(dist.items(),
                          key=lambda x: (float("inf") if x[0].startswith(">") else int(x[0].split("-")[0])))
    return dict(sorted_items)


# ----------------------
# Tag & Scene handling
# ----------------------
def suppress_tags(df, suppress_list):
    """Remove suppressed tags; drop rows with no tags left."""
    if "matched_tags" not in df.columns:
        return df
    new_rows = []
    for _, row in df.iterrows():
        tags = row["matched_tags"]
        if not isinstance(tags, dict):
            continue
        filtered_tags = {k: v for k, v in tags.items() if k not in suppress_list}
        if filtered_tags:
            row["matched_tags"] = filtered_tags
            new_rows.append(row)
    return pd.DataFrame(new_rows)


def categorize(df, category_file):
    """Categorize matched tags into scenes with matched_scene field."""
    with open(category_file, "r") as f:
        category_map = json.load(f)

    matched_scenes = []
    for tags in df["matched_tags"]:
        if not isinstance(tags, dict) or len(tags) == 0:
            matched_scenes.append([])
            continue

        scenes_for_row = []
        for scene, keywords in category_map.items():
            best_tag, best_score = None, -1.0
            for kw in keywords:
                if kw in tags and tags[kw] > best_score:
                    best_tag, best_score = kw, tags[kw]
            if best_tag:
                scenes_for_row.append({
                    "scene": scene,
                    "score": float(best_score),
                    "peak_tag": best_tag
                })
        matched_scenes.append(scenes_for_row)

    df["matched_scene"] = matched_scenes
    return df


def apply_top_k(df, top_k):
    """Keep only top_k rows per scene by highest scene score."""
    if "matched_scene" not in df.columns or not top_k:
        return df

    keep_indices = set()
    for scene in set(s["scene"] for row in df["matched_scene"] for s in row):
        scene_rows = []
        for idx, row in df.iterrows():
            for s in row["matched_scene"]:
                if s["scene"] == scene:
                    scene_rows.append((idx, s["score"]))
        scene_rows.sort(key=lambda x: x[1], reverse=True)
        keep_indices.update([idx for idx, _ in scene_rows[:top_k]])

    return df.loc[sorted(list(keep_indices))]


# ----------------------
# Save results
# ----------------------
def save_dataset(df, jsonl_file, parquet_file, output_dir, prefix, suffix):
    """Save dataset; use suffix to reflect cleaning process."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = prefix + "_" + suffix if suffix else prefix

    if jsonl_file:
        out = os.path.join(output_dir, prefix + ".jsonl")
        df.to_json(out, orient="records", lines=True)
        print(f"✅ Saved cleaned jsonl → {out}")
    elif parquet_file:
        out = os.path.join(output_dir, prefix + ".parquet")
        df.to_parquet(out, index=False)
        print(f"✅ Saved cleaned parquet → {out}")
    return prefix


# ----------------------
# Stats + Plots
# ----------------------
def generate_stats(df, output_dir, prefix, total_loaded, df_full, top_n=50):
    """Generate stats.json and visualization plots."""
    stats = {}
    stats["total_rows_loaded"] = total_loaded
    stats["rows_after_filter"] = len(df)

    # Tag stats
    tag_counter = Counter()
    for tags in df.get("matched_tags", []):
        if isinstance(tags, dict):
            tag_counter.update(tags.keys())
    stats["tags_total"] = len(tag_counter)
    stats["tag_match_counts"] = dict(tag_counter)
    stats["top_tags"] = tag_counter.most_common(top_n)

    # Scene stats
    if "matched_scene" in df.columns:
        scene_counter = Counter()
        for scenes in df["matched_scene"]:
            for s in scenes:
                scene_counter[s["scene"]] += 1
        stats["scene_total"] = len(scene_counter)
        stats["scene_counts"] = dict(scene_counter)

    # Duration stats — from full dataset before filtering
    if "duration" in df_full.columns:
        durations = df_full["duration"].dropna().tolist()
        if durations:
            dist = duration_distribution(df_full)
            stats["duration_stats_full"] = {
                "distribution": dist,
                "min": min(durations),
                "max": max(durations),
                "mean": float(sum(durations) / len(durations)),
                "median": float(sorted(durations)[len(durations)//2]),
            }

            # Plot duration distribution (only bins with data)
            labels, values = zip(*dist.items())
            nonzero = [(l, v) for l, v in zip(labels, values) if v > 0]
            labels, values = zip(*nonzero)
            plt.figure(figsize=(14, 6))
            plt.bar(labels, values, color="seagreen")
            plt.xticks(rotation=90)
            plt.title("Duration Distribution (All Data)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, prefix + "_duration_distribution.png"), dpi=200)
            plt.close()

    # Save stats.json
    stats_out = os.path.join(output_dir, prefix + "_stats.json")
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Stats saved to {stats_out}")

    # Plot tag distribution
    if tag_counter:
        tags, counts = zip(*tag_counter.most_common(top_n))
        plt.figure(figsize=(14, 6))
        plt.bar(tags, counts, color="steelblue")
        plt.xticks(rotation=75, ha="right")
        plt.title("Top Tag Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, prefix + "_tag_distribution.png"), dpi=200)
        plt.close()

    # Plot scene distribution
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


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Clean video dataset with suppress/categorize/top_k options.")
    parser.add_argument("--jsonl", type=str, help="Input JSONL file")
    parser.add_argument("--parquet", type=str, help="Input Parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for cleaned data & stats")
    parser.add_argument("--output_prefix", type=str, default=None, help="Prefix for output files (optional)")
    parser.add_argument("--suppress", type=str, nargs="*", default=["hand gesture", "playful gesture"],
                        help="List of tags to suppress (drop rows if no tags left)")
    parser.add_argument("--categorize", type=str, default=None,
                        help="JSON file mapping categories to tag lists")
    parser.add_argument("--top_k", type=int, default=None,
                        help="If set, keep only top_k rows per scene with highest scene_score")
    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.jsonl, args.parquet)
    total_loaded = len(df)

    # Extract duration upfront for ALL rows (df_full for stats)
    if "timestamp" in df.columns:
        df["duration"] = df["timestamp"].apply(extract_duration)
    elif "start" in df.columns and "end" in df.columns:
        df["duration"] = df["end"] - df["start"]
    else:
        df["duration"] = None
    df_full = df.copy()

    # Suppress tags
    if args.suppress:
        df = suppress_tags(df, args.suppress)

    # Categorize
    suffixes = []
    if args.categorize:
        df = categorize(df, args.categorize)
        suffixes.append("categorize")

    # Top_k
    if args.top_k:
        df = apply_top_k(df, args.top_k)
        suffixes.append(f"top{args.top_k}")

    # Output prefix
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        if args.jsonl:
            prefix = os.path.splitext(os.path.basename(args.jsonl))[0]
        else:
            prefix = os.path.splitext(os.path.basename(args.parquet))[0]

    suffix = "_".join(suffixes) if suffixes else "cleaned"
    prefix = save_dataset(df, args.jsonl, args.parquet, args.output_dir, prefix, suffix)

    # Stats + plots
    generate_stats(df, args.output_dir, prefix, total_loaded, df_full)


if __name__ == "__main__":
    main()