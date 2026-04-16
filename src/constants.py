from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

CURRENT_MODEL_TITLE = "qwen3-0.6b"
MODEL_REPOS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen-3.5-2b": "Qwen/Qwen-3.5-2B",
}


def get_model_repo(model_title: str = CURRENT_MODEL_TITLE) -> str:
    if model_title not in MODEL_REPOS:
        raise ValueError(
            f"Unknown model title '{model_title}'. Add it to MODEL_REPOS in constants.py."
        )
    return MODEL_REPOS[model_title]


def get_model_dir(model_title: str = CURRENT_MODEL_TITLE) -> Path:
    return MODELS_DIR / model_title

OPENR1_MATH = "open_r1_math_220k"
BESPOKE_STRATOS = "bespokelabs/Bespoke-Stratos-17k"
FILTERED_BESPOKE_STRATOS_DIR = "bespoke-stratos-17k-filtered"
LORA_DATA = "lora_data.jsonl"
RLVR_DATA = "rlvr_data.jsonl"
TEST_DATA = "test_data.jsonl"