#!/usr/bin/env python3
"""Aggregate step-level rollout JSONL into episode-level JSONL."""

import argparse
import json
import os
from collections import defaultdict

from agentdebug.data.episode import episode_group_key, step_rows_to_episode


def load_step_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert step-level rollout JSONL into episode-level JSONL")
    parser.add_argument("--input_jsonl", required=True, help="Path to step-level rollout JSONL")
    parser.add_argument("--output_jsonl", required=True, help="Path to write episode-level JSONL")
    args = parser.parse_args()

    grouped_rows: dict[tuple, list[dict]] = defaultdict(list)
    for row in load_step_rows(args.input_jsonl):
        grouped_rows[episode_group_key(row)].append(row)

    episodes = [step_rows_to_episode(rows) for _, rows in sorted(grouped_rows.items(), key=lambda item: item[0])]

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as file_obj:
        for episode in episodes:
            file_obj.write(json.dumps(episode, ensure_ascii=False) + "\n")

    print(f"Wrote {len(episodes)} episodes to {args.output_jsonl}")


if __name__ == "__main__":
    main()
