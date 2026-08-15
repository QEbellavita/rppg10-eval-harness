# Contributing

Thanks for helping improve reproducible physiological-signal evaluation.

## Good first contributions

- Improve documentation or examples without adding private data paths.
- Add synthetic tests for an edge case in extraction, scoring, or release-boundary checks.
- Improve error messages or command-line ergonomics.
- Review the evidence policy and flag claims that are not supported by a reproducible test.

## Before opening an issue

Please include the command, Python version, operating system, a minimal synthetic example
where possible, and the full error message. Do not attach raw videos, ECG files,
subject-linked rows, model weights, or licensed dataset material.

## Before opening a pull request

```bash
.venv/bin/python -m pytest tests/
```

Keep changes focused, preserve the release-boundary tests, and describe any claim that
depends on a licensed dataset as a local-only validation step.

## Scope

This repository is research infrastructure. It does not provide medical advice, diagnose
conditions, or grant access to external datasets and model weights.
