import json

Comments = r"..\Datasets\Filtered\RC_2024-07.jsonl"
Submissions = r"..\Datasets\Filtered\RS_2024-07.jsonl"

output = r"..\Datasets\Reddit_train.jsonl"

def load_dataset(filename, keep_keys):
    db={}
    count = 0
    print(f"Reading {filename}...")

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                lean_obj = {k: obj.get(k) for k in keep_keys}
                if 'parent_id' in obj:
                    lean_obj['parent_id'] = obj['parent_id']
                if 'link_id' in obj:
                    lean_obj['link_id'] = obj['link_id']
                db[obj['id']] = lean_obj
                count+=1

                if count%500000 == 0:
                    print(f"Loaded {count} rows in memory")
            except:
                continue

    print("Loaded Dataset in memory")
    return db

sub_db = load_dataset(Submissions, keep_keys=["title", "selftext"])
com_db = load_dataset(Comments, keep_keys=["body", "subreddit"])

print("All Datasets loaded")

t1_pairs = 0
t3_pairs = 0
processed = 0

batch=[]
with open(output, 'w', encoding='utf-8') as f:
    for com_id, comment in com_db.items():
        processed+=1
        parent_id = comment.get('parent_id', '')

        #Reply to a comment
        if parent_id.startswith('t1_'):
            parent_key = parent_id.split('_')[1]

            if parent_key in com_db:
                parent = com_db[parent_key]
                subreddit = comment['subreddit']
                instruction = f"You are a user in r/{subreddit}, Reply to the given comment."
                link = comment.get('link_id', '')
                if link.startswith("t3_"):
                    link_id = link.split('_')[1]
                    if link_id in sub_db:
                        submission = sub_db[link_id]
                        context = submission['title']
                        instruction = f"You are a user in r/{subreddit}, Reply to the given comment in context: {context}."
                input = parent['body']
                output = comment['body']
                entry = {
                    "instruction":instruction,
                    "input":input,
                    "output":output,
                    "subreddit":subreddit
                }
                batch.append(json.dumps(entry))
                t1_pairs+=1

        #Reply to a submission
        elif parent_id.startswith('t3'):
            parent_key = parent_id.split('_')[1]

            if parent_key in sub_db:
                parent = sub_db[parent_key]
                context = parent['title']
                output = comment['body']
                subreddit = comment['subreddit']
                input=""
                instruction=""
                if parent['selftext']:
                    input = parent['selftext']
                    instruction = f"You are a user in r/{subreddit}, Reply to the given comment in context: {context}."
                else:
                    input = parent['title']
                    instruction = f"You are a user in r/{subreddit}, Reply to the given comment."
                entry = {
                    "instruction":instruction,
                    "input":input,
                    "output":output,
                    "subreddit":subreddit
                }
                batch.append(json.dumps(entry))
                t3_pairs += 1

        if processed%100000 == 0:
            f.write('\n'.join(batch) + '\n')
            batch.clear()
            total = t1_pairs + t3_pairs
            print(f" Processed: {processed:,} and made {total:,} pairs.")

    if batch:
        f.write('\n'.join(batch) + '\n')

total_pairs = t1_pairs + t3_pairs
print(f"\n\n🎉 DONE! Generated {total_pairs:,} pairs.")
print(f"   - Paired Comment Replies: {t1_pairs:,}")
print(f"   - Paired Post Replies:    {t3_pairs:,}")
print(f"   - Saved to: {output}")