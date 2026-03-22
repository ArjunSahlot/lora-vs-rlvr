from huggingface_hub import snapshot_download

from constants import *

snapshot_download(
    repo_id="Qwen/Qwen-3.5-2B",
    local_dir=MODELS_DIR / QWEN2B,
)