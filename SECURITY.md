# Security Audit Report

This document records the results of a security audit of the NeuroCognition repository's git history, performed to identify any sensitive information (API keys, credentials, private infrastructure URLs) that may have been committed.

## Audit Summary

**Date:** 2026-03-03  
**Scope:** Full git commit history of the `reggans/NeuroCognition` repository  
**Current main branch status:** ✅ Clean — no hardcoded secrets

---

## Findings

### 1. Private LiteLLM Proxy Server URL (HIGH)

**Affected file:** `model_wrapper.py` (root-level, later moved to `shared/model_wrapper.py`)  
**Affected commits:** Commits from approximately 2025-03-25 through 2025-05-03  
**Example commit:** [`6c51410`](https://github.com/reggans/NeuroCognition/commit/6c51410c2a21c435a1c7fe6740cbd738f753170e)

**Description:**  
A private institutional LiteLLM proxy server URL was hardcoded in `model_wrapper.py`:

```python
self.client = openai.OpenAI(
    api_key=api_key,
    base_url = "REMOVED"
)
```

This URL identifies a private AI model proxy server belonging to an institutional lab. While no API key value was hardcoded (the key was correctly read from `LITELLM_API_KEY` environment variable), the server URL itself is sensitive infrastructure information.

**Status:** Resolved in current `main` branch — the LiteLLM source was replaced with OpenRouter and other configurable endpoints in `shared/model_wrapper.py`.

**Recommended action:** Rotate any `LITELLM_API_KEY` credentials that were used with this server, in case the URL exposure led to unauthorized access attempts.

---

### 2. Local Conda Environment Path (LOW)

**Affected file:** `environment.yml`  
**Affected commits:** Commits prior to [`04424b2`](https://github.com/reggans/NeuroCognition/commit/04424b29b85513cd2aea4f7c587ab0d73098a2f0)

**Description:**  
The conda environment file contained a local filesystem path that revealed the developer's username and home directory structure:

```yaml
prefix: /home/faeyza/miniconda3/envs/swm
```

**Status:** Resolved — removed in the "anonymize repo" commit `04424b2`.

---

## Remediation Steps Taken

1. **Current branch is clean:** All sensitive information has been removed from the current `main` branch.
2. **Added `gitleaks.toml`:** Configured secret scanning rules to detect API keys, private URLs, and other credentials in future commits.
3. **Added GitHub Actions workflow:** `.github/workflows/secret-scan.yml` runs gitleaks on every push and pull request, scanning the full commit history.

## Remaining Risk

The old commits containing the LiteLLM URL remain accessible in the public git history on GitHub. To fully eliminate this risk:

1. **Rotate credentials:** Rotate the `LITELLM_API_KEY` that was used with the `litellm.rum.uilab.kr` server.
2. **Consider history rewrite:** Use [BFG Repo Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) or `git filter-repo` to remove the sensitive commits from history, then force-push. This requires coordinating with all contributors.

```bash
# Example using BFG Repo Cleaner (requires repository owner action)
bfg --replace-text secrets.txt
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push origin --force --all
```

## Prevention

To prevent future secrets from being committed:

- **Never hardcode API keys, tokens, or private infrastructure URLs** in source code. Always use environment variables.
- The `gitleaks` scanner (configured in `gitleaks.toml`) will automatically flag secrets on future pull requests.
- Sensitive environment variable names used by this project:
  - `OPENAI_API_KEY` — OpenAI API key
  - `OPENROUTER_API_KEY` — OpenRouter API key  
  - `GEMINI_API_KEY` — Google Gemini API key
  - `GOOGLE_API_KEY` — Google API key (legacy)
  - `HF_TOKEN` — HuggingFace access token
  - `LITELLM_API_KEY` — LiteLLM proxy API key
