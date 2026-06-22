# Security Policy

## Reporting

Do not open a public issue for credential exposure or another vulnerability
that could put users at immediate risk. Contact the maintainers privately
through the security contact listed in the public repository profile.

## Credential Handling

- Configure API credentials only through environment variables.
- Keep `.env` files, logs, indexes, and experiment outputs out of Git.
- Run `python scripts/scan_secrets.py` before every release.
- Revoke and rotate a credential immediately if it is committed or shared.

The bundled scanner is intentionally small and only catches high-confidence
patterns. Use an established scanner such as Gitleaks in CI for public
repositories.
