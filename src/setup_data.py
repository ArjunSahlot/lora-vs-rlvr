import json
from datasets import load_dataset

from constants import *

TRAIN_SET = 5000
TEST_SET = 1000

ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
ds.save_to_disk(DATA_DIR / OPENR1_MATH)
subset = ds.shuffle(seed=0).select(range(TRAIN_SET + TEST_SET))

lora_data = []
rlvr_data = []
test_data = []

for i in range(TRAIN_SET):
    row = subset[i]
    
    problem = row["problem"]
    trace = row["solution"]
    solution = row["answer"]

    lora_data.append({
        "problem": problem,
        "trace": trace,
    })

    rlvr_data.append({
        "problem": problem,
        "solution": solution
    })

for i in range(TRAIN_SET, TRAIN_SET + TEST_SET):
    row = subset[i]
    
    problem = row["problem"]
    trace = row["solution"]
    solution = row["answer"]

    test_data.append({
        "problem": problem,
        "trace": trace,
        "solution": solution
    })

with open(DATA_DIR / LORA_DATA, "w") as f:
    for item in lora_data:
        f.write(json.dumps(item) + "\n")

with open(DATA_DIR / RLVR_DATA, "w") as f:
    for item in rlvr_data:
        f.write(json.dumps(item) + "\n")

with open(DATA_DIR / TEST_DATA, "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")
