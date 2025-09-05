#!/usr/bin/env python3
import argparse
import os, json, ast
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt


# ---------- I/O ----------
def load_dataset(jsonl_file=None, parquet_file=None):
    """Prefer JSONL if both given (no concatenation)."""
    if jsonl_file:
        return pd.read_json(jsonl_file, lines=True)
    if parquet_file:
        return pd.read_parquet(parquet_file)
    raise ValueError("At least one of --jsonl or --parquet must be provided.")


def save_dataset(df, jsonl_file, parquet_file, output_dir, prefix, suffix):
    """Save cleaned dataset in the same preferred format: JSONL > Parquet."""
    os.makedirs(output_dir, exist_ok=True)
    out_prefix = f"{prefix}_{suffix}" if suffix else prefix

    if jsonl_file:
        out = os.path.join(output_dir, out_prefix + ".jsonl")
        df.to_json(out, orient="records", lines=True)
        print(f"✅ Saved cleaned JSONL → {out}")
    elif parquet_file:
        out = os.path.join(output_dir, out_prefix + ".parquet")
        df.to_parquet(out, index=False)
        print(f"✅ Saved cleaned Parquet → {out}")
    else:
        # fallback: if only one path was omitted by caller, still save a JSONL
        out = os.path.join(output_dir, out_prefix + ".jsonl")
        df.to_json(out, orient="records", lines=True)
        print(f"✅ Saved cleaned JSONL → {out}")

    return out_prefix


# ---------- Helpers ----------
def ensure_dict_maybe_literal(x):
    """Normalize matched_tags / matched_scene field to a dict if possible."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        # Try parse as literal (e.g., "{'tag': 0.9}")
        try:
            v = ast.literal_eval(x)
            if isinstance(v, dict):
                return v
        except Exception:
            return None
    return None


def ensure_scene_list(x):
    """
    Normalize matched_scene to a list of {'scene': str, 'score': float} objects.
    Accepts:
      - dict: {scene: score, ...}
      - list of objects: [{'scene':..., 'score':...}, ...]
      - string of either format
    """
    if isinstance(x, dict):
        return [{"scene": k, "score": float(v)} for k, v in x.items()]
    if isinstance(x, list):
        # assume already list of objects
        out = []
        for it in x:
            if isinstance(it, dict) and "scene" in it and "score" in it:
                out.append({"scene": it["scene"], "score": float(it["score"])})
        return out
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return ensure_scene_list(v)
        except Exception:
            return []
    return []


def parse_time_to_seconds(time_str):
    from datetime import datetime as _dt
    try:
        t = _dt.strptime(time_str, "%H:%M:%S.%f")
    except ValueError:
        t = _dt.strptime(time_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second + (t.microsecond / 1e6)


def extract_duration_from_timestamp(ts_str):
    """timestamp like: "['0:01:38.348', '0:01:42.602']" → seconds (float)"""
    try:
        ts_list = ast.literal_eval(ts_str)
        if isinstance(ts_list, (list, tuple)) and len(ts_list) == 2:
            start = parse_time_to_seconds(ts_list[0])
            end = parse_time_to_seconds(ts_list[1])
            dur = end - start
            return dur if dur >= 0 else None
    except Exception:
        return None
    return None


# ---------- Cleaning steps ----------
def compute_duration_if_missing(df):
    if "duration" in df.columns and df["duration"].notnull().any():
        return df
    if "timestamp" in df.columns:
        df["duration"] = df["timestamp"].apply(extract_duration_from_timestamp)
    elif {"start", "end"}.issubset(df.columns):
        df["duration"] = df["end"] - df["start"]
    else:
        df["duration"] = None
    return df


def suppress_tags(df, suppress_list):
    """Drop suppressed tags from matched_tags; drop rows with no tags left."""
    if "matched_tags" not in df.columns:
        return df

    kept_rows = []
    for _, row in df.iterrows():
        tags = row["matched_tags"]
        if not isinstance(tags, dict):
            tags = ensure_dict_maybe_literal(tags)
        if not isinstance(tags, dict):
            continue

        if suppress_list:
            filtered = {k: v for k, v in tags.items() if k not in suppress_list}
        else:
            filtered = dict(tags)

        if filtered:
            row = row.copy()
            row["matched_tags"] = filtered
            kept_rows.append(row)

    return pd.DataFrame(kept_rows) if kept_rows else df.iloc[0:0]


def derive_matched_scene_from_mapping(df, category_json):
    """
    If matched_scene not present, derive it from matched_tags using scene→tags mapping.
    Take scene score as the max tag score among its member tags present in matched_tags.
    """
    with open(category_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)  # {scene: [tags...]}

    derived = []
    for _, row in df.iterrows():
        tags = row.get("matched_tags", {})
        if not isinstance(tags, dict):
            tags = ensure_dict_maybe_literal(tags)
        if not isinstance(tags, dict) or not tags:
            derived.append({})
            continue

        s2score = {}
        for scene, tag_list in mapping.items():
            best = None
            for t in tag_list:
                if t in tags:
                    score = float(tags[t])
                    best = score if (best is None or score > best) else best
            if best is not None:
                s2score[scene] = float(best)
        derived.append(s2score)

    df = df.copy()
    df["matched_scene"] = derived
    return df


def apply_top_k_per_scene(df, top_k):
    """
    Keep only rows that are in the top_k by scene score for each scene.
    Requires matched_scene present (dict or list of {scene,score}).
    """
    if not top_k or "matched_scene" not in df.columns or len(df) == 0:
        return df

    # Normalize matched_scene to dict for ranking
    scene_scores_per_row = []
    for _, row in df.iterrows():
        ms = row["matched_scene"]
        if isinstance(ms, dict):
            scene_scores_per_row.append(ms)
        else:
            ms_list = ensure_scene_list(ms)
            scene_scores_per_row.append({d["scene"]: float(d["score"]) for d in ms_list})

    # Build per-scene ranking
    per_scene_rows = defaultdict(list)  # scene → [(idx, score), ...]
    for idx, s2v in zip(df.index, scene_scores_per_row):
        for scene, score in s2v.items():
            per_scene_rows[scene].append((idx, float(score)))

    keep = set()
    for scene, rows in per_scene_rows.items():
        rows.sort(key=lambda x: x[1], reverse=True)
        keep.update(idx for idx, _ in rows[:top_k])

    # Filter to kept indices
    return df.loc[sorted(list(keep))]


# ---------- Stats & Viz ----------
def duration_distribution(df, bin_size=5, max_limit=300):
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
    # sort by start of bin numerically
    return dict(sorted(dist.items(),
                       key=lambda x: (float("inf") if x[0].startswith(">") else int(x[0].split("-")[0]))))


def plot_bar(name, data_pairs, out_png, title, rotate=75):
    if not data_pairs:
        return
    labels, counts = zip(*data_pairs)
    plt.figure(figsize=(14, 6))
    plt.bar(labels, counts)
    plt.xticks(rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def generate_stats_and_plots(df_clean, output_dir, prefix, total_loaded, top_n=50):
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_rows_loaded": int(total_loaded),
        "rows_after_filter": int(len(df_clean)),
    }

    # Tag stats (cleaned results)
    tag_counter = Counter()
    if "matched_tags" in df_clean.columns:
        for x in df_clean["matched_tags"]:
            d = ensure_dict_maybe_literal(x) if not isinstance(x, dict) else x
            if isinstance(d, dict):
                tag_counter.update(d.keys())
    stats["tags_total"] = len(tag_counter)
    stats["tag_match_counts"] = dict(tag_counter)
    stats["top_tags"] = tag_counter.most_common(top_n)

    # Scene stats (cleaned results)
    scene_counter = Counter()
    if "matched_scene" in df_clean.columns:
        for x in df_clean["matched_scene"]:
            if isinstance(x, dict):
                scene_counter.update(x.keys())
            else:
                for it in ensure_scene_list(x):
                    scene_counter.update([it["scene"]])
    stats["scene_total"] = len(scene_counter)
    stats["scene_counts"] = dict(scene_counter)

    # Duration stats (cleaned results)
    df_clean = compute_duration_if_missing(df_clean)
    dist = duration_distribution(df_clean)
    durations = df_clean["duration"].dropna().tolist()
    stats["duration_stats"] = {
        "distribution": dist,
        "min": float(min(durations)) if durations else None,
        "max": float(max(durations)) if durations else None,
        "mean": float(sum(durations) / len(durations)) if durations else None,
        "median": float(sorted(durations)[len(durations)//2]) if durations else None,
    }

    # Save stats
    os.makedirs(output_dir, exist_ok=True)
    stats_out = os.path.join(output_dir, prefix + "_stats.json")
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"📝 Stats saved → {stats_out}")

    # Plots (cleaned results)
    # Tags
    tag_pairs = Counter(stats["tag_match_counts"]).most_common(top_n)
    plot_bar("tags", tag_pairs, os.path.join(output_dir, prefix + "_tag_distribution.png"),
             "Top Tag Distribution", rotate=75)
    # Scenes
    scene_pairs = sorted(stats["scene_counts"].items(), key=lambda x: x[0])
    plot_bar("scenes", scene_pairs, os.path.join(output_dir, prefix + "_scene_distribution.png"),
             "Scene Distribution", rotate=45)
    # Duration
    # Trim to non-zero bins for readability
    dur_pairs = [(k, v) for k, v in dist.items() if v > 0]
    plot_bar("duration", dur_pairs, os.path.join(output_dir, prefix + "_duration_distribution.png"),
             "Duration Distribution", rotate=90)

    return stats_out


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Data cleaner for filtered video data (supports suppress, top_k, optional categorize)."
    )
    parser.add_argument("--jsonl", type=str, help="Input JSONL file (preferred if both given)")
    parser.add_argument("--parquet", type=str, help="Input Parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for cleaned data & stats")
    parser.add_argument("--output_prefix", type=str, default=None, help="Prefix for output files (optional)")

    parser.add_argument("--suppress", type=str, nargs="*", default=None,
                        help="List of tags to suppress; rows with no tags left are dropped.")
    parser.add_argument("--categorize", type=str, default=None,
                        help="Optional JSON mapping {scene: [tags,...]} to derive matched_scene if missing.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="If set, keep only top_k rows per scene by scene score (requires matched_scene or --categorize).")

    args = parser.parse_args()

    # Load
    df = load_dataset(args.jsonl, args.parquet)
    total_loaded = len(df)
    print(f"📦 Loaded {total_loaded} rows")

    # Ensure duration exists for plotting later
    df = compute_duration_if_missing(df)

    # Suppress tags
    suffixes = []
    if args.suppress:
        df = suppress_tags(df, args.suppress)
        suffixes.append("suppress")

    # Ensure matched_scene exists if we need top_k and it's missing
    need_scene = args.top_k and args.top_k > 0
    if need_scene and "matched_scene" not in df.columns:
        if not args.categorize:
            raise ValueError("--top_k requires matched_scene or --categorize JSON to derive it.")
        df = derive_matched_scene_from_mapping(df, args.categorize)
        suffixes.append("categorize")

    # Apply top_k per scene (if requested)
    if args.top_k:
        before = len(df)
        df = apply_top_k_per_scene(df, args.top_k)
        print(f"🏆 top_k={args.top_k}: kept {len(df)}/{before} rows")
        suffixes.append(f"top{args.top_k}")

    # Decide output prefix
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        if args.jsonl:
            prefix = os.path.splitext(os.path.basename(args.jsonl))[0]
        elif args.parquet:
            prefix = os.path.splitext(os.path.basename(args.parquet))[0]
        else:
            prefix = "cleaned"

    suffix = "_".join(suffixes) if suffixes else "cleaned"
    out_prefix = save_dataset(df, args.jsonl, args.parquet, args.output_dir, prefix, suffix)

    # Stats & plots on CLEANED results
    stats_path = generate_stats_and_plots(df, args.output_dir, out_prefix, total_loaded)
    print(f"✅ Done. Stats at: {stats_path}")


if __name__ == "__main__":
    main()