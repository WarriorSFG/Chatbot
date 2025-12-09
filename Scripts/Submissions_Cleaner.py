import zstandard as zstd
import io
import json
import re

from langdetect import detect

DATABASE = r"..\Datasets\submissions\RS_2024-07.zst"
FILTERED = r"..\Datasets\Filtered\RS_2024-07.jsonl"

KEEP_KEYS = { "id", "subreddit",  "title", "selftext","score"}
CommonBotsList = {
    "AutoModerator",
    "RemindMeBot",
    "WikiTextBot",
    "ClickableLinkBot",
    "HCE_Replacement_Bot",
    "BookStoreBot",
    "haiku_bot",
    "AmputatorBot",
    "Shakespeare-Bot",
    "Reddit-Book-Bot",
    "SaveVideo",
    "SaveVideobot",
    "gifreversingbot",
    "wandering-dwarf-miner",
    "CommunityModBot"
}

def squeeze_repeats(text):
    return re.sub(r'(.)\1{3,}', r'\1\1\1', text)

def isenglish(text):
    try:
        if text.isascii():
            return True
        return detect(text) == 'en'
    except:
        return False

def is_all_caps(text):
    letters = sum(1 for c in text if c.isalpha())
    caps = sum(1 for c in text if c.isupper())
    return letters > 0 and (caps / letters) ==1

def contains_links(text):
    return "http://" in text or "https://" in text or "www." in text

def verify_submission(obj):
    title = obj.get("title", "[removed]").strip()
    body = obj.get("selftext", "").strip()
    author = obj.get("author", "bot")
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
    if not isenglish(combined):
        return False
    if '![gif]' in body:
        return False
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
                    title = minimal.get("title", "")
                    body = minimal.get("selftext", "")
                    if body:
                        body = squeeze_repeats(body)
                        body = body.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;"," ")
                        body = re.split(r'(?i)\n\s*edit:', body)[0]
                        body = re.sub(r"^>.*$", "", body, flags=re.MULTILINE)
                        body = re.sub(r"<[^>]+>", "", body)
                        minimal["selftext"] = body.strip()
                    title = squeeze_repeats(title)
                    title = re.split(r'(?i)\n\s*edit:', title)[0]
                    title = re.sub(r"^>.*$", "", title, flags=re.MULTILINE)
                    title = re.sub(r"<[^>]+>", "", title)
                    if len(title) < 10:
                        continue
                    minimal["title"] = title.strip()
                    minimal["subreddit"] = minimal.get("subreddit") or "AskReddit"

                    batch.append(json.dumps(minimal))
                    count += 1

                    if count % 100000 == 0:
                        f.write("\n".join(batch) + "\n")
                        print(f"Processed {count} valid submissions...")
                        batch.clear()

                except json.JSONDecodeError:
                    continue

            if batch:
                f.write("\n".join(batch) + "\n")

print("Submission filtering complete.")

