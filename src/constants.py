from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

QWEN2B = "qwen3-0.6b"

OPENR1_MATH = "open_r1_math_220k"
LORA_DATA = "lora_data.jsonl"
RLVR_DATA = "rlvr_data.jsonl"
TEST_DATA = "test_data.jsonl"