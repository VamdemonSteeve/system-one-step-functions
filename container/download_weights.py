"""Build-time step: pull the Laya checkpoint into HF_HOME so the image is self-contained."""

import os

import laya

repo = os.environ.get("LAYA_REPO", "convaiinnovations/laya")
subfolder = os.environ.get("LAYA_SUBFOLDER") or None

kwargs = {"subfolder": subfolder} if subfolder else {}
laya.load(repo, **kwargs)
print(f"Cached {repo}{'/' + subfolder if subfolder else ''} into {os.environ.get('HF_HOME')}")
