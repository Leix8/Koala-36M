#!/usr/bin/env python3
import argparse, os, subprocess, sys, shlex
from datetime import datetime

def build_shard_filenames(output_dir, output_prefix, shard_index, num_shards):
    width = len(str(num_shards))
    base = f"{output_prefix}_shard{shard_index:0{width}d}_of{num_shards}"
    parquet_out = os.path.join(output_dir, f"{base}.parquet")
    jsonl_out   = os.path.join(output_dir, f"{base}.jsonl")
    log_json    = os.path.join(output_dir, f"{base}_log.json")
    text_log    = os.path.join(output_dir, "logs", f"{base}.log")
    return parquet_out, jsonl_out, log_json, text_log


def main():
    parser = argparse.ArgumentParser(
        description="Sequential shard runner that launches each shard with Accelerate (multi-GPU within each shard).",
        allow_abbrev=False
    )

    # Orchestrator-only args
    parser.add_argument("--worker", default="semantic_tag_filtering.py",
                        help="Path to the worker script.")
    parser.add_argument("--num_shards", type=int, default = 1000,
                        help="Total number of shards.")
    parser.add_argument("--start", type=int, default=0,
                        help="First shard index to process (inclusive).")
    parser.add_argument("--end", type=int, default=None,
                        help="Last shard index to process (inclusive). Defaults to num_shards-1.")
    parser.add_argument("--gpu_ids", type=str, default=('0,1,2,3,4,5'),
                        help="Comma-separated GPU IDs (e.g. '0,1,2,3'). If omitted, uses all visible GPUs.")
    parser.add_argument("--output_dir", type=str, default = "./filtered",
                        help="Output directory (must match worker's).")
    parser.add_argument("--output_prefix", type=str, default = "koala-36m",
                        help="Output prefix (must match worker's).")
    parser.add_argument("--accelerate-mixed-precision", default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Accelerate mixed precision mode.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a shard if its parquet already exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing.")

    # Parse only known args → everything else goes to worker
    args, worker_args = parser.parse_known_args()

    start = args.start
    end = args.end if args.end is not None else args.num_shards - 1
    assert 0 <= start <= end < args.num_shards, "Invalid shard range"

    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    accel_cmd = [
        "python", "-m", "accelerate.commands.launch",
        "--multi_gpu",
        f"--mixed_precision={args.accelerate_mixed_precision}",
    ]

    env = os.environ.copy()
    if args.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    succeeded, failed = [], []

    for idx in range(start, end + 1):
        parquet_out, jsonl_out, log_json, text_log = build_shard_filenames(
            args.output_dir, args.output_prefix, idx, args.num_shards
        )

        if args.skip_existing and os.path.exists(parquet_out):
            print(f"[SKIP] shard {idx}: {parquet_out} exists")
            succeeded.append(idx)
            continue

        worker_cmd = [
            args.worker,
            "--shard_index", str(idx),
            "--num_shards", str(args.num_shards),
            "--output_dir", args.output_dir,
            "--output_prefix", args.output_prefix,
        ] + worker_args

        cmd = accel_cmd + worker_cmd

        print(f"\n=== Running shard {idx}/{args.num_shards-1} ===")
        print("CMD:", " ".join(shlex.quote(c) for c in cmd))
        print("LOG:", text_log)

        if args.dry_run:
            continue

        os.makedirs(os.path.dirname(text_log), exist_ok=True)

        with open(text_log, "w") as logf:
            logf.write(f"# Launched at {datetime.now().isoformat()}\n")
            logf.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
            logf.flush()
            ret = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            ok = (ret.returncode == 0)

        if ok and os.path.exists(parquet_out):
            print(f"[OK] shard {idx}")
            succeeded.append(idx)
        else:
            print(f"[FAIL] shard {idx} (see log {text_log})")
            failed.append(idx)

    print("\n=== SUMMARY ===")
    print(f"Succeeded shards: {succeeded}")
    print(f"Failed shards   : {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()