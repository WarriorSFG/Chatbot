import os
import json
import random

TEMP = r"../../Datasets/Temp.jsonl"
INPUT = r"../../Datasets/Reddit_train.jsonl"
OUTPUT_FILE = r"../../Datasets/Train/Reddit.jsonl"
BATCH_SIZE = 50_000

def load_rows(file, limit=None):
    rows = []
    count = 0
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if limit and count >= limit: break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
                count+=1
            except json.JSONDecodeError:
                continue
    return rows

def write_shuffled(rows, output_path):
    print("Shuffling data...")
    random.shuffle(rows)

    print(f" Writing output to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

data = load_rows(INPUT)
write_shuffled(data, TEMP)
data = load_rows(TEMP, 500_000)
write_shuffled(data, OUTPUT_FILE)