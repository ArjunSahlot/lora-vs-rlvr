import argparse
import time

import torch
from unsloth import FastLanguageModel

from constants import CURRENT_MODEL_TITLE, get_model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=CURRENT_MODEL_TITLE, help="Model title to test")
    parser.add_argument(
        "--prompt",
        default="Write a hello world in Rust.",
        help="Prompt for quick generation test",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(get_model_dir(args.model)),
        max_seq_length=2048,
    )
    FastLanguageModel.for_inference(model)

    try:
        messages = [{"role": "user", "content": args.prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    except TypeError:
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

    prompt_tokens = int(inputs.shape[-1])
    start = time.perf_counter()
    outputs = model.generate(input_ids=inputs, max_new_tokens=args.max_new_tokens)
    elapsed = time.perf_counter() - start

    generated_tokens = int(outputs.shape[-1]) - prompt_tokens
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    generated_text = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Latency: {elapsed:.3f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
    print("\nResponse:\n")
    print(generated_text)


if __name__ == "__main__":
    main()