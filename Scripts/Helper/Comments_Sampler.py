import zstandard as zstd
import io
import json

database = "..\..\Datasets\comments\RC_2024-07.zst"
SampledSet = "..\..\SampledSet\comments\RC_2024-07.jsonl"

with open(database, 'rb') as db:
    dctx = zstd.ZstdDecompressor()

    with open(SampledSet, 'w+', encoding='utf-8') as f:
        with dctx.stream_reader(db) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            count = 0
            for line in text_stream:
                f.write(line)
                if count == 2048:
                    break
                else:
                    count+=1

print("Sampling complete.")
