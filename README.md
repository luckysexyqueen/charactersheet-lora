---
title: CharacterSheet Triple-LoRA
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
---

# CharacterSheet Triple-LoRA — guarded auto-recovery

This package preserves the previous app and portable operations, and adds `auto_repair.py`.

```bash
python auto_repair.py --root .
python auto_repair.py --root . --repair-safe
python -m unittest discover -s tests -v
```

Checks: required project files, Python syntax, Git availability, HF_TOKEN presence, PyTorch and CUDA presence. Produces `reports/latest_diagnostics.json` in UTF-8. Safe repair only creates `outputs/` and `reports/`, never overwrites app source, changes dependencies, runs untrusted code, downloads arbitrary executables, or bypasses access restrictions.

For verified downloads use `python operations.py clone --url https://huggingface.co/OWNER/REPO --dest ./models/repo` with legitimate access. Upload requires a write-capable `HF_TOKEN` and `python operations.py upload ...` explicitly. GitHub/GitLab/npm/PyPI are potential sources of trusted pinned packages, **not** automatically executed repair scripts. Source code patches require tests and approval; keep version-control backups for rollback.

No script can guarantee 100% uptime, unlimited free GPU, permanent storage, internet connectivity, or compatibility among three LoRA weights. Hugging Face free Spaces may sleep or restart; persistent outputs require external storage or manual backups.

## 자동 진단 및 안전 복구 (신규)

`python auto_repair.py --repair`는 Python 구문, 필수 파일, GPU/PyTorch, 토큰, Git 상태를 점검하고 작업 폴더를 복구합니다. `reports/latest-health.json`에 결과를 저장합니다. `--strict`는 발견된 문제가 있으면 비정상 종료합니다.

GitHub Actions의 `.github/workflows/health-check.yml`은 매일 점검과 테스트를 수행합니다. **자동으로 인터넷의 임의 코드를 설치하거나 실행하지 않고, Space의 원본 소스를 무단 수정하거나 외부 저장소로 업로드하지 않습니다.** 모델 인증, GPU 할당, 인터넷 서비스 장애, 비호환 LoRA는 사람이 해결해야 할 수 있습니다. 100% 성공이나 평생 무중단은 보장할 수 없습니다. 자동 수정은 안전한 디렉터리 복구로 제한되며, 의존성·모델 업그레이드는 검토 후 적용하세요.


## Resilience patch
`python resilience.py check` checks source syntax and installed dependencies.
`python resilience.py backup` creates checksum-manifest source backup.
`python resilience.py restore backups/source-....zip` verifies and restores the backup; creates a backup of current files first.
`python resilience.py lookup pypi gradio` fetches metadata only from allowlisted PyPI/npm/Hugging Face endpoints, with timeouts, retries and size limits. No downloaded code is executed or installed automatically.
Automatic deployment and 100% repair success cannot be guaranteed; GPU quota, licensing, authentication and service availability are external dependencies.
