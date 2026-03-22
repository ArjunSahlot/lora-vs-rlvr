from unsloth import FastLanguageModel
import torch

model_path = "./qwen-3.5-2b"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    # load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "Write a hello world in Rust."}
        ]
    }
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 64)
print(tokenizer.batch_decode(outputs)[0])