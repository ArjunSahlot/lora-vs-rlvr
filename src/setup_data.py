import json
from datasets import load_dataset

from constants import *

SUBSET_COUNT = 5000

ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
ds.save_to_disk(DATA_DIR / OPENR1_MATH)
subset = ds.shuffle(seed=0).select(range(SUBSET_COUNT))

lora_data = []
rlvr_data = []

for row in subset:
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

with open(DATA_DIR / LORA_DATA, "w") as f:
    for item in lora_data:
        f.write(json.dumps(item) + "\n")

with open(DATA_DIR / RLVR_DATA, "w") as f:
    for item in rlvr_data:
        f.write(json.dumps(item) + "\n")
