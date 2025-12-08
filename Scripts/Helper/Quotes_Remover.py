import json
import os
import re
import gc
import time

Filtered = r"..\..\Datasets\Filtered\RC_2024-07.jsonl"
temp = r"..\..\Datasets\temp.jsonl"

def polish_text(text):
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r'!\[gif\]\(.*?\)', '', text)
    return text.strip()

print(f"🧹 Polishing {Filtered}...")

cleaned = 0
dropped = 0
count = 0

batch=[]
try:
    with open(Filtered, 'r', encoding='utf-8') as db:
        with open(temp, 'w', encoding='utf-8') as f:
            for line in db:
                try:
                    obj = json.loads(line)
                    original_body = obj['body']
                    polished_body = polish_text(original_body)

                    if len(polished_body) < 15:
                        dropped+=1
                        continue
                    if original_body != polished_body:
                        cleaned+=1
                    obj['body'] = polished_body
                    batch.append(json.dumps(obj))
                    count+=1
                    if count%100000 == 0:
                        print(f"processed {count:,} rows")
                        f.write('\n'.join(batch) + '\n')
                        batch.clear()
                except json.JSONDecodeError as e:
                    print(e)
                    continue

            if batch:
                f.write('\n'.join(batch) + '\n')
except Exception as e:
    print(f"\n❌ Error: {e}")

gc.collect()
time.sleep(1)
try:
    print(f"\n✨ Done! Cleaned: {cleaned:,} | Dropped: {dropped:,} Total: {count:,}")
    print("Cleaning up leftover files...")
    if os.path.exists(Filtered):
        os.remove(Filtered)
        os.rename(temp, Filtered)
        print(f"✅ Success! {Filtered} has been updated.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Your original file is safe. The temp file is:", temp)
