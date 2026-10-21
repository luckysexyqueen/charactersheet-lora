"""Portable and authorized project operations; no hosting-limit bypass."""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def parse_repo_url(url):
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != "huggingface.co" or u.query or u.fragment:
        raise ValueError("Only HTTPS huggingface.co repository URLs are supported")
    parts = [p for p in u.path.strip("/").split("/") if p]
    if len(parts) == 3 and parts[0] in ("spaces", "datasets"):
        return ("space" if parts[0] == "spaces" else "dataset"), "/".join(parts[1:])
    if len(parts) == 2:
        return "model", "/".join(parts)
    raise ValueError("Expected a model, dataset, or Space repository root URL")


def backup_outputs(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Output folder does not exist: {root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        for f in root.rglob("*"):
            if f.is_file() and not f.is_symlink() and f.resolve() != destination:
                z.write(f, f.relative_to(root))
    return destination


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            p = Path(name)
            if p.is_absolute() or ".." in p.parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe ZIP member: {name}")
            target = (destination / p).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe ZIP member: {name}")
        destination.mkdir(parents=True, exist_ok=True)
        z.extractall(destination)
    return destination


def doctor():
    print("Python:", sys.version.split()[0])
    print("Git:", shutil.which("git") or "not installed")
    print("HF_TOKEN:", "configured" if os.getenv("HF_TOKEN") else "not configured")
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
    except ImportError:
        print("PyTorch: not installed")
    print("Output folder:", os.getenv("OUTPUT_ROOT", "./outputs"))
    print("Note: 9B image inference needs compatible GPU and model/LoRA access.")


def clone(url, destination):
    repo_type, repo_id = parse_repo_url(url)
    # huggingface_hub handles authenticated and public downloads using standard permissions.
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=destination,
                             token=os.getenv("HF_TOKEN") or None)


def upload(folder, repo_id, repo_type):
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN with write permission is required")
    if repo_type not in ("space", "model", "dataset"):
        raise ValueError("Invalid repository type")
    return HfApi(token=token).upload_folder(folder_path=str(folder), repo_id=repo_id,
                                              repo_type=repo_type, commit_message="Update CharacterSheet project")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    a=sub.add_parser("backup");a.add_argument("--source",default="./outputs");a.add_argument("--out",required=True)
    a=sub.add_parser("restore");a.add_argument("--archive",required=True);a.add_argument("--dest",required=True)
    a=sub.add_parser("clone");a.add_argument("--url",required=True);a.add_argument("--dest",required=True)
    a=sub.add_parser("upload");a.add_argument("--folder",default=".");a.add_argument("--repo",required=True);a.add_argument("--type",default="space",choices=["space","model","dataset"])
    args=parser.parse_args()
    if args.cmd=="doctor":doctor()
    elif args.cmd=="backup":print(backup_outputs(args.source,args.out))
    elif args.cmd=="restore":print(safe_extract(args.archive,args.dest))
    elif args.cmd=="clone":print(clone(args.url,args.dest))
    elif args.cmd=="upload":print(upload(args.folder,args.repo,args.type))

if __name__=="__main__":main()
