import os
import json
import random


DATASETS_DIR = r"../Datasets/Train"
OUTPUT_FILE = r"../Datasets/Train/final_training.jsonl"
VALID_EXTENSIONS = (".jsonl",)
BATCH_SIZE = 100_000

def find_jsonl_files(base_dir):
    jsonl_files = []
    if os.path.exists(base_dir):
        for file in os.listdir(base_dir):
            if file.endswith(VALID_EXTENSIONS):
                jsonl_files.append(os.path.join(base_dir, file))
    return jsonl_files

def load_all_rows(files):
    rows = []
    for file_path in files:
        print(f"Loading: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
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

print(" Starting Dataset Merger")
files = find_jsonl_files(DATASETS_DIR)

if not files:
  print(" No JSONL files found!")
else:
    print(f"Found {len(files)} dataset files")
    all_rows = load_all_rows(files)
    print(f" Total rows loaded: {len(all_rows)}")
    write_shuffled(all_rows, OUTPUT_FILE)
    print(" Dataset merge complete.")

