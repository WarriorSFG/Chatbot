import zstandard as zstd
import io
import json
import re
DATABASE = "..\\..\\Datasets\\submissions\\RS_2024-07.zst"
FILTERED = "..\\..\\Datasets\\Filtered\\RS_2024-07.jsonl"

KEEP_KEYS = { "id", "subreddit",  "title", "selftext","score","author"}
CommonBotsList = {"AutoModerator","RemindMeBot","WikiTextBot","ClickableLinkBot",
"HCE_Replacement_Bot","BookStoreBot","haiku_bot","AmputatorBot","Shakespeare-Bot","Reddit-Book-Bot","SaveVideo","SaveVideobot","gifreversingbot","wandering-dwarf-miner","CommunityModBot"}
def is_all_caps(text):
    letters = sum(1 for c in text if c.isalpha())
    caps = sum(1 for c in text if c.isupper())
    return letters > 0 and (caps / letters) ==1
def contains_links(text):
    return "http://" in text or "https://" in text or "www." in text
def verify_submission(obj):
    title = obj.get("title", "").strip()
    body = obj.get("selftext", "").strip()
    author = obj.get("author", "").lower()
    score = obj.get("score", 0)
    # Deleted / removed
    if title in ("[deleted]", "[removed]"):
        return 1==0

    if author.lower().endswith("bot"):
        return 1==0
    if author in CommonBotsList:
        return 1==0
    if score <50:
        return 1==0

    if not (10 <= len(title) <= 200):
        return 1==0
    if is_all_caps(title):
        return 1==0
    if body:
        if not (20 <= len(body) <= 1000):
            return 1==0
    else:
        # body empty 
        if len(title) < 40:
            return 1==0

    combined = f"{title} {body}"
    if contains_links(combined):
        return 1==0
    return 1==1

# -------- EXTRACTION --------
print("🚀 Starting Submission Extraction...")

with open(DATABASE, "rb") as db:
    dctx = zstd.ZstdDecompressor()
    batch = []

    with open(FILTERED, "w", encoding="utf-8") as f:
        with dctx.stream_reader(db) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            count = 0
            for line in text_stream:
                try:
                    obj = json.loads(line)

                    if not verify_submission(obj):
                        continue

                    minimal = {key: obj.get(key) for key in KEEP_KEYS}
                    minimal["subreddit"] = minimal.get("subreddit") or "AskReddit"
                    minimal["title"] = minimal["title"].strip()
                    minimal["selftext"] = minimal.get("selftext", "").strip()

                    batch.append(json.dumps(minimal, ensure_ascii=False))
                    count += 1

                    if count % 10000 == 0:
                        f.write("\n".join(batch) + "\n")
                        print(f"Processed {count} valid submissions...", end="\r")
                        batch.clear()

                except json.JSONDecodeError:
                    continue

            if batch:
                f.write("\n".join(batch) + "\n")

print("Submission filtering complete.")

