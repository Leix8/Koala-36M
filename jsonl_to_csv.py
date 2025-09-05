#!/usr/bin/env python3
import argparse
import pandas as pd
import json
import os


def flatten_jsonl_to_csv(input_file, output_file=None):
    """
    Load a JSONL file and flatten its contents into a CSV file.
    Converts matched_tags and matched_scene into compact string fields.
    """
    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # Flatten matched_tags into a single string column
            tags = obj.get("matched_tags", {})
            if isinstance(tags, dict):
                tag_strs = [f"{k.replace(' ', '_')}:{round(v,4)}" for k, v in tags.items()]
                obj["tag"] = "; ".join(tag_strs) if tag_strs else None
            obj.pop("matched_tags", None)

            # Flatten matched_scene into a single string column
            scenes = obj.get("matched_scene", {})
            if isinstance(scenes, dict):
                scene_strs = [f"{k}:{round(v,4)}" for k, v in scenes.items()]
                obj["scene"] = "; ".join(scene_strs) if scene_strs else None
            obj.pop("matched_scene", None)

            rows.append(obj)

    df = pd.DataFrame(rows)

    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".csv"

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ CSV saved → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL dataset into flat CSV")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Optional path to output CSV file")
    args = parser.parse_args()

    flatten_jsonl_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()