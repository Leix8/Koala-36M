import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import json
import sys

def load_dataset_auto(path, sample=None):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        if sample:  # ✅ Only grab first `sample` rows
            ds = load_dataset(path, split="train", streaming=True)
            rows = []
            for i, row in enumerate(ds):
                rows.append(row)
                if i + 1 >= sample:
                    break
            df = pd.DataFrame(rows)
        else:
            ds = load_dataset(path, split="train")
            df = ds.to_pandas()
    return df

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
    """Return list of tags matched in caption (semantic or exact)."""
    if semantic:
        caption_emb = model.encode(
            caption, convert_to_tensor=True, normalize_embeddings=True
        )
        cos_scores = util.cos_sim(caption_emb, tag_embeddings)[0]
        return [tags[i] for i, score in enumerate(cos_scores) if score >= threshold]
    else:
        # Simple exact substring matching
        return [tag for tag in tags if tag.lower() in caption.lower()]


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
    parser.add_argument("--semantic", action="store_true",
                        help="Use semantic embeddings (default: off → exact matching).")

    args = parser.parse_args()

    # 1. Load dataset
    df = load_dataset_auto(args.input, sample=args.sample)
    print(f"Loaded dataset with {len(df)} rows from {args.input}")

    if args.sample:
        df = df.sample(args.sample, random_state=42)

    print("loading dataset finished")

    # 2. Load tags
    tags = load_tags(args.tags)
    print(f"Loaded {len(tags)} tags: {tags}")

    # 3. Load model if semantic mode
    if args.semantic:
        print(f"Loading model: {args.model}")
        model = SentenceTransformer(args.model)
        tag_embeddings = model.encode(tags, convert_to_tensor=True, normalize_embeddings=True)
    else:
        model, tag_embeddings = None, None
    
    print("loading tags finished")


    # 4. Apply filtering
    
    df["matched_tags"] = df["caption"].apply(
        lambda cap: find_tags(cap, tags, tag_embeddings, model,
                              threshold=args.threshold, semantic=args.semantic)
    )

    filtered = df[df["matched_tags"].map(len) > 0]

    # 5. Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Build output paths
    parquet_out = os.path.join(args.output_dir, f"{args.output_prefix}.parquet")
    jsonl_out   = os.path.join(args.output_dir, f"{args.output_prefix}.jsonl")

    # Save results
    filtered.to_parquet(parquet_out, index=False)
    filtered.to_json(jsonl_out, orient="records", lines=True)

    print(f"✅ Filtered dataset contains {len(filtered)} rows out of {len(df)}")
    print(f"Saved to: {parquet_out}, {jsonl_out}")



if __name__ == "__main__":
    main()

