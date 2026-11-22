# Changelog

All notable changes to **forensic-signature-ai** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-22

### Added
- Modular package facade under `src/` (`src/__init__.py`, `src/forensic_signature.py`) with LRU cached threshold profile and structured logging.
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`) executing Ruff lint, Pytest test suite, Docker build, Semgrep SAST, and Trivy container vulnerability scanning.
- Pinned dependencies in `requirements.txt`, `requirements-lock.txt`, `uv.lock`, and Ruff config (`ruff.toml`).
- Security and governance artifacts: `.github/dependabot.yml`, `SECURITY.md`, `Makefile`, `runbook.md`, `LICENSE`, `CONTRIBUTING.md`, and `.pre-commit-config.yaml`.
