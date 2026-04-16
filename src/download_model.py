from huggingface_hub import snapshot_download

from constants import *

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=MODELS_DIR / QWEN2B,
)