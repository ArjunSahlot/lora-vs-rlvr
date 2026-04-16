import json
import argparse
from datasets import load_dataset

from constants import (
    BESPOKE_STRATOS,
    DATA_DIR,
    FILTERED_BESPOKE_STRATOS_DIR,
    LORA_DATA,
    RLVR_DATA,
    TEST_DATA,
)

TRAIN_SET_DEFAULT = 5000
TEST_SET_DEFAULT = 1000

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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=TRAIN_SET_DEFAULT)
    parser.add_argument("--test-size", type=int, default=TEST_SET_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    total_set = args.train_size + args.test_size

    ds = load_dataset(BESPOKE_STRATOS, split="train")
    filtered_ds = ds.filter(filter_fn)
    subset = filtered_ds.shuffle(seed=args.seed).select(range(total_set))
    filtered_ds.save_to_disk(DATA_DIR / FILTERED_BESPOKE_STRATOS_DIR)

    lora_data = []
    rlvr_data = []
    test_data = []

    for i in range(total_set):
        row = subset[i]

        conversations = row["conversations"]
        problem = conversations[0]["value"]
        trace = conversations[1]["value"]

        solution = get_answer(trace)

        if i < args.train_size:
            lora_data.append({"problem": problem, "trace": trace})
            rlvr_data.append({"problem": problem, "solution": solution})
        else:
            test_data.append({"problem": problem, "trace": trace, "solution": solution})

    with open(DATA_DIR / LORA_DATA, "w", encoding="utf-8") as f:
        for item in lora_data:
            f.write(json.dumps(item) + "\n")

    with open(DATA_DIR / RLVR_DATA, "w", encoding="utf-8") as f:
        for item in rlvr_data:
            f.write(json.dumps(item) + "\n")

    with open(DATA_DIR / TEST_DATA, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()