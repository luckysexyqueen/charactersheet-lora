"""Safe, bounded diagnostics, backup and dependency discovery. No remote code execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from importlib import metadata

ROOT_FILES = ('app.py', 'auto_repair.py', 'operations.py', 'requirements.txt')
SOURCES = {
    'pypi': 'https://pypi.org/pypi/{package}/json',
    'npm': 'https://registry.npmjs.org/{package}',
    'huggingface': 'https://huggingface.co/api/models/{package}',
}

def diagnose(root):
    root = Path(root).resolve()
    issues = []
    for name in ROOT_FILES:
        path = root / name
        if not path.is_file():
            issues.append({'code': 'missing_file', 'file': name})
        elif name.endswith('.py'):
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
            except (SyntaxError, UnicodeError) as exc:
                issues.append({'code': 'syntax_error', 'file': name, 'detail': str(exc)})
    if shutil.which('git') is None:
        issues.append({'code': 'git_unavailable'})
    if not os.getenv('HF_TOKEN'):
        issues.append({'code': 'hf_token_missing', 'severity': 'notice'})
    for package in ('gradio', 'torch', 'diffusers', 'huggingface_hub'):
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            issues.append({'code': 'dependency_missing', 'package': package})
    return {'timestamp': datetime.now(timezone.utc).isoformat(), 'issues': issues,
            'status': 'needs_attention' if issues else 'healthy'}

def backup(root):
    root = Path(root).resolve()
    target = root / 'backups'
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = target / ('source-' + stamp + '.zip')
    manifest = {}
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ROOT_FILES:
            path = root / name
            if path.is_file():
                data = path.read_bytes()
                z.writestr(name, data)
                manifest[name] = hashlib.sha256(data).hexdigest()
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
    return str(dest)

def restore(root, archive):
    root = Path(root).resolve()
    archive = Path(archive).resolve()
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('manifest.json'))
        if not manifest or not set(manifest).issubset(ROOT_FILES):
            raise ValueError('Invalid backup manifest')
        payload = {}
        for name, expected in manifest.items():
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Backup checksum mismatch: ' + name)
            if name.endswith('.py'):
                ast.parse(data.decode('utf-8-sig'))
            payload[name] = data
    # Save the current state before applying verified backup.
    backup(root)
    for name, data in payload.items():
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
            tmp.write(data)
            tempname = tmp.name
        os.replace(tempname, root / name)
    return sorted(payload)

def fetch_metadata(source, package, timeout=6, attempts=2):
    if source not in SOURCES:
        raise ValueError('Unapproved source')
    from urllib.parse import quote
    url = SOURCES[source].format(package=quote(package, safe='/@'))
    req = urllib.request.Request(url, headers={'User-Agent': 'CharacterSheet-healthcheck/1.0', 'Accept': 'application/json'})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024:
                    raise ValueError('Metadata response too large')
                return {'source': source, 'package': package, 'url': url,
                        'metadata': json.loads(data.decode('utf-8'))}
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('check')
    sub.add_parser('backup')
    restore_parser = sub.add_parser('restore')
    restore_parser.add_argument('archive')
    lookup = sub.add_parser('lookup')
    lookup.add_argument('source', choices=sorted(SOURCES))
    lookup.add_argument('package')
    args = parser.parse_args()
    if args.action == 'check':
        result = diagnose(args.root)
    elif args.action == 'backup':
        result = {'backup': backup(args.root)}
    elif args.action == 'restore':
        result = {'restored': restore(args.root, args.archive)}
    else:
        result = fetch_metadata(args.source, args.package)
        result['metadata'] = {'available': True, 'keys': list(result['metadata'])[:15]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
