from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen-3.5-2B",
    local_dir="./qwen-3.5-2b",
)