from datasets import load_dataset
import json
import os

output_directory = r"..\Datasets"

def save_to_json(data, filename):
    path = os.path.join(output_directory,filename)
    print(f"Saving {len(data):,} rows to {filename}...")
    with open(path, 'w',encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def ds_formatter(name, outputfile_name, instruction, limit=None):
    print(f"\n📦 Processing {outputfile_name}...")
    try:
        ds = load_dataset(name,split='train')
        formatted_data =[]
        count = 0
        for row in ds:
            if limit and count >= limit:
                break

            formatted_data.append({
                "instruction": instruction,
                "input": row['instruction'],
                "output": row['output']
            })
            count+=1
        save_to_json(formatted_data, f"{outputfile_name}.jsonl")
        print(f"Saved {outputfile_name}!")
    except Exception as e:
        print(f"Error saving {name} dataset.: {e}")

ds_formatter('nickrosh/Evol-Instruct-Code-80k-v1', "Evol_Code","You are a Senior Python Developer. Provide a comprehensive solution with code and explanation.", limit=180000)

print("\n📦 Processing Python Syntax (Alpaca Version)...")
try:
    ds = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
    formatted_data = []

    for row in ds:
        dataset_inst = row.get('instruction', '')
        dataset_inp = row.get('input', '')

        full_input = f"{dataset_inst}\n{dataset_inp}".strip()

        formatted_data.append({
            "instruction": "You are a Python Expert. Write a Python script to solve this. Return only the code.",
            "input": full_input,
            "output": row['output']
        })

    save_to_json(formatted_data, "Python_Syntax.jsonl")
    print("✅ Saved Python_Syntax.jsonl!")
except Exception as e:
    print(f"❌ Error saving Python Syntax: {e}")

print("\n📦 Processing Math PoT (Calculator)...")

ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
count = 0
formatted_data =[]
orcaset = []
target_count = 100000
for row in ds:
    if count >= target_count: break
    inp = row['instruction']
    out = row['output'].lower()
    if 'write a program' in out or ('python' in out and 'program' in out):
        formatted_data.append({
            "instruction": "You are a Calculator. Write a python script to solve this. Don't talk. Just output <TOOL_CALL> code.",
            "input": inp,
            "output": f"<TOOL_CALL>\n{row['output']}\n</TOOL_CALL>"
        })
        count += 1
    else:
        orcaset.append({
            "instruction": "You are a Math Tutor. Explain the steps. Don't code. Use logic and words to explain how.",
            "input": inp,
            "output": row['output']
        })
save_to_json(formatted_data, "Math_PoT.jsonl")

print("\n📦 Processing Math CoT (Tutor)...")
ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
formatted_data = list(orcaset)
print(f"   (Inherited {len(formatted_data):,} rows from MathInstruct overflow)")
orcaset.clear()
target_count = 100000
count = 0

for row in ds:
    if count >= target_count: break
    formatted_data.append({
        "instruction": "You are a Math Tutor. Explain the steps. Don't code. Use logic and words to explain how.",
        "input": row['question'],
        "output": row['answer']
    })
    count += 1
save_to_json(formatted_data, "Math_CoT.jsonl")