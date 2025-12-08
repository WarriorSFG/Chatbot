from multiprocessing.reduction import duplicate

import zstandard as zstd
import io
import json

databaseC = "..\..\Datasets\comments\RC_2024-07.zst"
HeadersC =  "..\..\SampledSet\comments\Headers.txt"


databaseS = "..\..\Datasets\submissions\RS_2024-07.zst"
HeadersS =  "..\..\SampledSet\submissions\Headers.txt"

Comment_Headers = []
Submission_Headers = []

with open(databaseC, 'rb') as db:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(db) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')

        for i in range(2048):
            line = next(text_stream)
            data = json.loads(line)
            for item in list(data.keys()):
                if item in Comment_Headers:
                    pass
                else:
                    Comment_Headers.append(item)

print(Comment_Headers)

with open(databaseS, 'rb') as db:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(db) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')

        for i in range(2048):
            line = next(text_stream)
            data = json.loads(line)
            for item in list(data.keys()):
                if item in Submission_Headers:
                    pass
                else:
                    Submission_Headers.append(item)

print(Submission_Headers)

with open(HeadersS, 'w+') as f:
    for item in Submission_Headers:
        f.write(item + '\n')

countC = 0
countS = 0
with open(databaseC, 'rb') as db:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(db) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        for line in text_stream:
            countC +=1

with open(databaseS, 'rb') as db:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(db) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        for line in text_stream:
            countS +=1


duplicate = 0
for item in Submission_Headers:
    if item in Comment_Headers:
        duplicate +=1

print("Total comment fields :", len(Comment_Headers))
print("Total submission fields :", len(Submission_Headers))
print("Total duplicate fields :", duplicate)
print("Total comments :", countC)
print("Total submissions :", countS)
print("Extraction complete.")