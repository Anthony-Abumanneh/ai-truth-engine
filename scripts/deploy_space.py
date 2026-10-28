"""Deploy the web app to a Hugging Face Space (Docker SDK, free CPU tier).

    HF_TOKEN=hf_... python scripts/deploy_space.py <username>/<space-name>

Uploads only git-tracked files (never .env, local models or data), adds the
Space configuration header to its README, and lets Hugging Face build the
Dockerfile (models are baked in at build time; ~15-25 min first build).
Optional Space secrets (Settings -> Variables and secrets): TAVILY_API_KEY,
NCBI_API_KEY for live search; TE_ALLOW_LIVE=0 to turn live search off.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_HEADER = """---
title: AI Truth Engine
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Checks science and health claims against the literature
---

"""


def main() -> int:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        print(__doc__)
        return 2
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN (a write token from https://huggingface.co/settings/tokens).")
        return 2
    from huggingface_hub import HfApi
    space = sys.argv[1]
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    with tempfile.TemporaryDirectory() as tmp:
        for f in tracked:
            if f.startswith((".github/", "training/", "extension/", "docs/spec/")):
                continue                                  # not needed to run the app
            dest = Path(tmp) / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dest)
        readme = Path(tmp) / "README.md"
        readme.write_text(SPACE_HEADER + readme.read_text())
        api = HfApi(token=token)
        api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                          commit_message="Deploy AI Truth Engine")
    print(f"Uploaded. Build progress: https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
