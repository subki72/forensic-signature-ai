# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.3.x   | :white_check_mark: |
| < 2.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in **Forensic Signature AI**:

1. **Do NOT open a public GitHub issue** for undisclosed vulnerabilities.
2. Submit a private advisory via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`).
3. Include affected files/endpoints, reproduction steps, and severity estimate.

## Security Controls

- **Input & File Upload Hardening**: Strict 5 MB upload cap (`413`), zero-byte guard (`400`), and binary Magic Byte inspection (JPEG/PNG/WEBP, `415`) to prevent MIME spoofing.
- **Rate Limiting**: `slowapi` enforces 10 requests/minute per IP on `/verify` (`HTTP 429`).
- **Reduced Production Surface**: Interactive Swagger/ReDoc routes are disabled when `DEBUG=False`.
- **Automated Scanning**: Weekly Dependabot alerts, Semgrep SAST, and Trivy container image scanning run in CI.
