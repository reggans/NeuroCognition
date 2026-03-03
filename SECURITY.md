# Security — Sensitive Data in Git History

This document records sensitive data discovered in the repository's git history and the steps needed to fully remediate it.

> **Current `main` branch is clean.** All secrets listed below exist only in old commits and must be purged via a history rewrite.

---

## Findings

### 1. OpenAI API Key — `api_calc.ipynb` (CRITICAL)

| Field | Value |
|---|---|
| File | `api_calc.ipynb` |
| First introduced | commit `e632ade3` ("cards", 2025-08-08) |
| Last present | commit `ef14d126` ("Merge image branch into main", 2025-08-11) — file removed from tree by this commit |
| Type | OpenAI project API key (`sk-proj-…`) |
| Exposure | Key was hardcoded in a source cell used to call the OpenAI Usage API for org `UILab` (`org-1p5YaTt9pETJtTo4ziKYao3U`) |

**Immediate action required:** rotate / revoke this key at https://platform.openai.com/api-keys before any remediation.

---

### 2. Private LiteLLM Proxy URL — `model_wrapper.py`

| Field | Value |
|---|---|
| File | `model_wrapper.py` |
| Commits | `6c51410` → `301cf1c` (2025-03-25 → 2025-05-03, ~25 commits) |
| Type | Hardcoded internal server URL (`REMOVED`) |
| Exposure | Institutional AI proxy server address exposed; no API key values were hardcoded |

---

### 3. Conda Environment Path — `environment.yml`

| Field | Value |
|---|---|
| File | `environment.yml` |
| Commit | before `04424b2` ("anonymize repo", 2026-01-29) |
| Type | Local home directory path (`/home/faeyza/miniconda3/envs/swm`) |
| Status | **Removed** by the "anonymize repo" commit (already clean on `main`) |

---

## Remediation — Rewriting Git History

Because these secrets are embedded in old commits, adding them to `.gitignore` or deleting the files in a new commit is **not sufficient** — the old commits remain publicly readable.

The only complete fix is to **rewrite history and force-push**. The repo owner must do this locally.

### Option A — BFG Repo Cleaner (recommended)

```bash
# 1. Clone a fresh mirror
git clone --mirror https://github.com/reggans/NeuroCognition.git

# 2. Download BFG: https://rtyley.github.io/bfg-repo-cleaner/

# 3. Delete the notebook file from all history
java -jar bfg.jar --delete-files api_calc.ipynb NeuroCognition.git

# 4. Remove the hardcoded URL string from all blobs
java -jar bfg.jar --replace-text <(echo 'REMOVED==>REMOVED') NeuroCognition.git

# 5. Expire and pack
cd NeuroCognition.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 6. Force-push (coordinate with all collaborators first!)
git push --force
```

### Option B — `git filter-repo`

```bash
pip install git-filter-repo

git clone https://github.com/reggans/NeuroCognition.git
cd NeuroCognition

# Remove file from all history
git filter-repo --path api_calc.ipynb --invert-paths

# Replace the URL in all blobs
git filter-repo --replace-text <(echo 'REMOVED==>REMOVED')

# Force-push
git push origin --force --all
git push origin --force --tags
```

### After force-pushing

1. All collaborators must re-clone or run `git fetch --all && git reset --hard origin/main`.
2. Ask GitHub Support to purge cached views: https://support.github.com/
3. Verify with `git log --all -S "sk-proj-"` — should return no results.
