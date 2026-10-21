"""Conservative self-diagnostics and reversible local repairs; no arbitrary code execution."""
import argparse
import ast
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ('app.py', 'requirements.txt')

def inspect_project(root):
    root = Path(root)
    issues = []
    for name in REQUIRED:
        if not (root / name).is_file():
            issues.append({'code': 'MISSING_' + name.upper().replace('.', '_'), 'detail': name})
    if not (root / 'requirements.txt').is_file():
        issues.append({'code': 'MISSING_REQUIREMENTS', 'detail': 'requirements.txt is absent'})
    if (root / 'app.py').is_file():
        try:
            ast.parse((root / 'app.py').read_text(encoding='utf-8-sig'))
        except (SyntaxError, UnicodeError) as e:
            issues.append({'code': 'SYNTAX_ERROR', 'detail': str(e)})
    if not os.getenv('HF_TOKEN'):
        issues.append({'code': 'HF_TOKEN_UNSET', 'detail': 'Restricted models/uploads may require a token'})
    if shutil.which('git') is None:
        issues.append({'code': 'GIT_UNAVAILABLE', 'detail': 'Repository operations need git'})
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append({'code': 'GPU_UNAVAILABLE', 'detail': 'FLUX 9B generation requires adequate acceleration'})
    except ImportError:
        issues.append({'code': 'TORCH_UNAVAILABLE', 'detail': 'PyTorch not installed'})
    return issues

def apply_safe_fixes(root):
    root = Path(root)
    actions = []
    for name in ('outputs', 'reports', 'backups'):
        path = root / name
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            actions.append(f'created {name}/')
    return {'actions': actions, 'note': 'No application source or dependency versions were changed.'}

def report(root, repair=False):
    root = Path(root).resolve()
    actions = apply_safe_fixes(root)['actions'] if repair else []
    issues = inspect_project(root)
    data = {'timestamp': datetime.now(timezone.utc).isoformat(), 'issues': issues, 'actions': actions,
            'status': 'needs_attention' if issues else 'healthy'}
    if repair:
        dest = root / 'reports' / 'latest-health.json'
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return data

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--repair', action='store_true', help='Only create missing work directories')
    p.add_argument('--strict', action='store_true', help='Exit nonzero if issues detected')
    args = p.parse_args()
    data = report(args.root, args.repair)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 1 if args.strict and data['issues'] else 0

if __name__ == '__main__':
    sys.exit(main())
