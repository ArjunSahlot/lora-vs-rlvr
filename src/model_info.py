from unsloth import FastLanguageModel

model, _ = FastLanguageModel.from_pretrained(
    "./qwen-3.5-2b-local",
    load_in_4bit=True,
    local_files_only=True
)

print(model)