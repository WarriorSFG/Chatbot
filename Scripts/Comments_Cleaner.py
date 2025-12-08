import zstandard as zstd
import io
import json
import re
from langdetect import detect, LangDetectException

database = r"..\Datasets\comments\RC_2024-07.zst"
Filtered = r"..\Datasets\Filtered\RC_2024-07.jsonl"

KEEP_KEYS = {'id', 'parent_id', 'link_id', 'subreddit', 'body', 'score'}

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

def isenglish(text):
    try:
        if text.isascii():
            return True
        return detect(text) == 'en'
    except:
        return False

def squeeze_repeats(text):
    return re.sub(r'(.)\1{3,}', r'\1\1\1', text)

def verify_usability(obj):
    score = obj.get('score', 0)
    body = obj.get('body', '[deleted]')
    author = obj.get('author', 'bot')
    if body in ['[removed]', '[deleted]']:
        return False
    if score < 50:
        return  False
    if author in CommonBotsList or author.lower().endswith('bot'):
        return False
    if len(body) < 15 or len(body) > 1000:
        return False
    if 'http' in body or 'www.' in body:
        return False
    if '![gif]' in body:
        return False
    if not isenglish(body):
        return False
    return True

print("🚀 Starting Extraction...")
with open(database, 'rb') as db:
    dctx = zstd.ZstdDecompressor()
    obj_batch = []
    with open(Filtered, 'w+', encoding='utf-8') as f:
        with dctx.stream_reader(db) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            count = 0
            process = 0
            for line in text_stream:
                try:
                    obj = json.loads(line)
                    process+=1
                    if process%1000000 == 0:
                        print(f"process {process} rows")
                    if not verify_usability(obj):
                        continue
                    body = obj.get('body', '')
                    body = re.split(r'(?i)\n\s*edit:', body)[0]
                    body = re.sub(r"^>.*$", "", body, flags=re.MULTILINE)
                    body = re.sub(r"<[^>]+>", "", body)
                    subreddit = obj.get('subreddit', 'AskReddit')
                    score = obj.get('score', 0)
                    body = squeeze_repeats(body)
                    body = body.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;"," ")
                    minimal_obj = {key: obj.get(key) for key in KEEP_KEYS}
                    minimal_obj['body'] = body.strip()
                    obj_batch.append(json.dumps(minimal_obj))
                    count += 1
                    if count % 100000 == 0:
                        f.write('\n'.join(obj_batch) + '\n')
                        print(f"Added {count} valid rows...")
                        obj_batch.clear()
                except json.JSONDecodeError as e:
                    print(e)
                    continue

            if obj_batch:
                f.write('\n'.join(obj_batch) + '\n')

print("Filtered the entire dataset complete.")
