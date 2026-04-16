from unsloth import FastLanguageModel

from constants import CURRENT_MODEL_TITLE, get_model_dir


def main() -> None:
    model, _ = FastLanguageModel.from_pretrained(
        model_name=get_model_dir(CURRENT_MODEL_TITLE),
        load_in_4bit=True,
        local_files_only=True,
    )

    print(f"Loaded model title: {CURRENT_MODEL_TITLE}")
    print(model)


if __name__ == "__main__":
    main()