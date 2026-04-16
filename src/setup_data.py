import json
from datasets import load_dataset

from constants import *

TRAIN_SET = 5000
TEST_SET = 1000
TOTAL_SET = TRAIN_SET + TEST_SET

def get_answer(text):
    last_boxed_idx = text.rfind("\\boxed{")
    if last_boxed_idx == -1:
        return text
    
    content = ""
    brace_count = 0
    start_idx = last_boxed_idx + 7
    
    for char in text[start_idx:]:
        if char == "{":
            brace_count += 1
        elif char == "}":
            if brace_count == 0:
                break
            brace_count -= 1
        content += char
        
    return content

def filter_fn(example):
    conversations = example.get("conversations", [])
    if not conversations or len(conversations) < 2:
        return False
    
    prompt = conversations[0].get("value", "").strip()
    return prompt.startswith("Return your final response within \\boxed{}.")

ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
filtered_ds = ds.filter(filter_fn)
subset = filtered_ds.shuffle(seed=0).select(range(TOTAL_SET))
filtered_ds.save_to_disk(DATA_DIR / "bespoke-stratos-17k-filtered")

lora_data = []
rlvr_data = []
test_data = []

for i in range(TOTAL_SET):
    row = subset[i]
    
    conversations = row["conversations"]
    problem = conversations[0]["value"]
    trace = conversations[1]["value"]
    
    solution = get_answer(trace)

    if i < TRAIN_SET:
        lora_data.append({
            "problem": problem,
            "trace": trace,
        })

        rlvr_data.append({
            "problem": problem,
            "solution": solution
        })
    else:
        test_data.append({
            "problem": problem,
            "trace": trace,
            "solution": solution
        })

with open(DATA_DIR / LORA_DATA, "w", encoding="utf-8") as f:
    for item in lora_data:
        f.write(json.dumps(item) + "\n")

with open(DATA_DIR / RLVR_DATA, "w", encoding="utf-8") as f:
    for item in rlvr_data:
        f.write(json.dumps(item) + "\n")

with open(DATA_DIR / TEST_DATA, "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")