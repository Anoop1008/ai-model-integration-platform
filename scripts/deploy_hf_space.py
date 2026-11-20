"""Deploy this app to a Hugging Face Docker Space: the same Dockerfile and image as on Render.

One-time setup (deploy tooling only, not an app dependency):
    pip install huggingface_hub
    hf auth login            # paste a token with "write" access from huggingface.co/settings/tokens

Run:
    python scripts/deploy_hf_space.py                      # -> <your-username>/churn-inference-api
    python scripts/deploy_hf_space.py someone/other-name

Uploads only what the image needs, in one commit (one Space build).
"""

import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ["Dockerfile", ".dockerignore", "requirements.txt", "app/**/*", "artifacts/*", "benchmarks/results/*", "evaluation/results/*"]
GITHUB = "https://github.com/Anoop1008/ai-model-integration-platform"

# Spaces reads its settings from the README front matter. app_port matches the Dockerfile default.
SPACE_README = f"""---
title: Churn Inference API
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: TensorFlow churn model behind a FastAPI inference API
---

TensorFlow churn classifier served by FastAPI: SQLite prediction log, PSI drift indicator,
API-key auth (off on this demo) and rate limiting. Same Docker image as the Render deployment.

Source, documentation and measured results: {GITHUB}
"""


def files_to_upload() -> list[Path]:
    found = {f for pattern in PATTERNS for f in ROOT.glob(pattern) if f.is_file() and "__pycache__" not in f.parts}
    return sorted(found)


def main() -> None:
    api = HfApi()
    repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{api.whoami()['name']}/churn-inference-api"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    # Hugging Face shows the app inside a frame on huggingface.co; the page's CSP must allow that.
    api.add_space_variable(repo_id, "FRAME_ANCESTORS", "https://huggingface.co")

    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=SPACE_README.encode())]
    ops += [CommitOperationAdd(path_in_repo=f.relative_to(ROOT).as_posix(), path_or_fileobj=str(f)) for f in files_to_upload()]
    assert any(op.path_in_repo == "artifacts/churn_model.keras" for op in ops), "model file missing from upload"
    api.create_commit(repo_id, repo_type="space", operations=ops, commit_message="Deploy from GitHub main")

    user, name = repo_id.split("/")
    print(f"Space page: https://huggingface.co/spaces/{repo_id}")
    print(f"Direct app: https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
