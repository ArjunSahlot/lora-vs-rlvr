from unsloth import FastLanguageModel

from constants import *

model, _ = FastLanguageModel.from_pretrained(
    model_name = MODELS_DIR / QWEN2B,
    load_in_4bit=True,
    local_files_only=True
)

print(model)