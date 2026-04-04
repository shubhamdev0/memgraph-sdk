# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Memgraph AI, please report it responsibly.

**Email:** security@memgraph.ai

**Do NOT:**
- Open a public GitHub issue for security vulnerabilities
- Post details on social media before the fix is released

**We will:**
- Acknowledge your report within 48 hours
- Provide a fix timeline within 7 days
- Credit you in the security advisory (if desired)

## Supported Versions

| Version | Supported |
|---|---|
| 0.6.x | ✅ Current |
| < 0.6 | ❌ Upgrade recommended |

## Security Best Practices

- Never commit API keys to version control
- Use environment variables for `MEMGRAPH_API_KEY`
- Rotate API keys periodically via the dashboard
- Use tenant isolation for multi-user deployments
