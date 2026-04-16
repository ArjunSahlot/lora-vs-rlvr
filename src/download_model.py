import argparse

from huggingface_hub import snapshot_download

from constants import CURRENT_MODEL_TITLE, get_model_dir, get_model_repo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=CURRENT_MODEL_TITLE,
        help="Model title to download (must exist in MODEL_REPOS).",
    )
    args = parser.parse_args()

    model_title = args.model
    model_repo = get_model_repo(model_title)
    model_dir = get_model_dir(model_title)

    snapshot_download(
        repo_id=model_repo,
        local_dir=model_dir,
    )

    print(f"Downloaded '{model_repo}' to '{model_dir}'.")


if __name__ == "__main__":
    main()