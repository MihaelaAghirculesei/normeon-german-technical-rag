# Security Policy

This is a personal learning project and is **not** production-hardened. Do
not run it against untrusted input or expose it on a public network without
your own review. A full threat model is planned for a later milestone
(`docs/THREAT-MODEL.md`).

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue:

- Open a [private security advisory](https://github.com/MihaelaAghirculesei/normeon-german-technical-rag/security/advisories/new), or
- Email **aghirculesei@gmail.com** with the details and, if possible, a
  minimal reproduction.

You can expect an acknowledgement within a few days. Since this is a
solo, unpaid project there is no formal SLA, but confirmed issues will be
fixed on `main` and credited in the release notes unless you prefer to
stay anonymous.

## Scope

In scope: the backend application code in `backend/`, the CI workflow, and
the container/compose setup.

Out of scope: the bundled corpus documents in `corpus/` (public German
regulations and synthetic sample specs — see `corpus/manifest.yaml`), and
any third-party model or service the app is configured to call.
