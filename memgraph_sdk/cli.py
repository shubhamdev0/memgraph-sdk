#!/usr/bin/env python3
"""
Memgraph CLI - One-command setup for AI agent memory integration.

Usage:
    memgraph init       # Set up Memgraph in current project
    memgraph remember   # Store a memory
    memgraph recall     # Search memories
    memgraph status     # Check connection
"""
import argparse
import os
import shutil
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

# --- Constants ---
CLOUD_URL = "https://api.memgraph.ai/v1"
LOCAL_URL = os.getenv("MEMGRAPH_API_URL", "http://localhost:8001/v1")
CONFIG_FILE = ".memgraph.env"
SKILL_DIR = ".agent/skills/memgraph"

# Template for skill file
SKILL_TEMPLATE = '''---
name: Memgraph Memory
description: Seamless long-term memory for AI agents via Memgraph. Auto-learn, inject context, and suggest based on project history.
---

# Memgraph Memory Skill

This skill gives you persistent memory across sessions.

## MCP Tools Available

| Tool | Use For |
|------|---------|
| `memgraph_search` | Find past decisions, architecture, context |
| `memgraph_remember` | Store new facts and learnings |

## When to SEARCH
Call `memgraph_search` BEFORE answering questions about:
- Architecture or design patterns
- Previous decisions or bugs
- "How does X work?" / "Why did we choose Y?"

## When to REMEMBER
Call `memgraph_remember` AFTER:
- Completing a significant feature or fix
- User makes an architectural decision
- Resolving a tricky bug

Categories: `decision`, `architecture`, `bug_fix`, `preference`, `general`
'''

MCP_CONFIG_TEMPLATE = '''{
  "mcpServers": {
    "memgraph": {
      "command": "python3",
      "args": ["${MCP_SERVER_PATH}"],
      "env": {
        "MEMGRAPH_API_URL": "${API_URL}",
        "MEMGRAPH_TENANT_ID": "${TENANT_ID}",
        "MEMGRAPH_API_KEY": "${API_KEY}"
      }
    }
  }
}
'''


def get_package_dir():
    """Get the memgraph-sdk package installation directory."""
    return Path(__file__).parent.parent


def init_project(non_interactive: bool = False):
    """Initialize Memgraph in the current project.

    In non-interactive mode, reads config from environment variables:
    - MEMGRAPH_API_URL (default: cloud)
    - MEMGRAPH_TENANT_ID (required)
    - MEMGRAPH_API_KEY (optional)
    """
    print("🚀 Memgraph Agent Memory Setup\n")

    if non_interactive:
        # Non-interactive: read from env vars (CI/CD, Docker, scripts)
        api_url = os.getenv("MEMGRAPH_API_URL", CLOUD_URL)
        tenant_id = os.getenv("MEMGRAPH_TENANT_ID", "")
        api_key = os.getenv("MEMGRAPH_API_KEY", "")

        if not tenant_id:
            print("❌ MEMGRAPH_TENANT_ID environment variable is required in non-interactive mode.")
            return
    else:
        # Interactive: prompt user
        print("Where is your Memgraph server running?")
        print("  1) Cloud (api.memgraph.ai)")
        print("  2) On-prem / Local (localhost:8001)")
        print("  3) Custom URL")

        choice = input("\nChoose [1/2/3]: ").strip()

        if choice == "1":
            api_url = CLOUD_URL
        elif choice == "2":
            api_url = LOCAL_URL
        elif choice == "3":
            api_url = input("Enter API URL: ").strip()
        else:
            print("Invalid choice")
            return

        # 2. Get credentials
        tenant_id = input("\nTenant ID: ").strip()
        api_key = input("API Key (optional, press Enter to skip): ").strip()

    # 3. Create .memgraph.env
    config_path = Path(CONFIG_FILE)
    with open(config_path, "w") as f:
        f.write(f"MEMGRAPH_API_URL={api_url}\n")
        f.write(f"MEMGRAPH_TENANT_ID={tenant_id}\n")
        if api_key:
            f.write(f"MEMGRAPH_API_KEY={api_key}\n")
    print(f"\n✅ Created {CONFIG_FILE}")

    # 4. Create skill directory
    skill_path = Path(SKILL_DIR)
    skill_path.mkdir(parents=True, exist_ok=True)

    skill_file = skill_path / "SKILL.md"
    with open(skill_file, "w") as f:
        f.write(SKILL_TEMPLATE)
    print(f"✅ Created {skill_file}")

    # 5. Copy or reference MCP server
    # Try multiple paths: installed package, dev checkout, bundled
    mcp_candidates = [
        get_package_dir() / "examples" / "integrations" / "mcp_server.py",
        Path(__file__).parent / "mcp_server.py",
        get_package_dir() / "integrations" / "mcp_server.py",
    ]
    mcp_copied = False
    for mcp_server_path in mcp_candidates:
        if mcp_server_path.exists():
            mcp_dest = skill_path / "mcp_server.py"
            shutil.copy(mcp_server_path, mcp_dest)
            print(f"✅ Copied MCP server to {mcp_dest}")
            mcp_copied = True
            break

    if not mcp_copied:
        print("⚠️  MCP server template not found. You can create one manually.")

    # 6. Create MCP config hints
    print("\n📋 To enable MCP in your editor:")
    print("\n   For Cursor, add to .cursor/mcp.json:")
    mcp_config = MCP_CONFIG_TEMPLATE.replace("${API_URL}", api_url)
    mcp_config = mcp_config.replace("${TENANT_ID}", tenant_id)
    mcp_config = mcp_config.replace("${API_KEY}", api_key or "")
    mcp_config = mcp_config.replace("${MCP_SERVER_PATH}", str(skill_path / "mcp_server.py"))
    print(mcp_config)

    print("\n🎉 Memgraph agent memory is ready!")
    print("   Your AI agent can now use memgraph_search and memgraph_remember tools.")


def remember_cmd(text: str, category: str = "general"):
    """Store a memory via CLI."""
    if not requests:
        print("Error: 'requests' package not installed. Run: pip install requests")
        return

    config = load_config()
    if not config:
        print("Error: Not initialized. Run 'memgraph init' first.")
        return

    try:
        resp = requests.post(
            f"{config['api_url']}/ingest",
            data={
                "tenant_id": config["tenant_id"],
                "user_id": "cli_user",
                "text": f"[{category}] {text}",
            },
            headers={"X-API-KEY": config.get("api_key", "")},
            timeout=10
        )
        if resp.status_code == 200:
            print(f"✅ Remembered: {text[:80]}...")
        else:
            print(f"❌ Error: {resp.text}")
    except requests.ConnectionError:
        print("❌ Cannot connect to Memgraph server")


def recall_cmd(query: str):
    """Search memories via CLI."""
    if not requests:
        print("Error: 'requests' package not installed. Run: pip install requests")
        return

    config = load_config()
    if not config:
        print("Error: Not initialized. Run 'memgraph init' first.")
        return

    try:
        resp = requests.post(
            f"{config['api_url']}/context",
            json={
                "tenant_id": config["tenant_id"],
                "user_id": "cli_user",
                "task": query,
                "agent_id": "cli"
            },
            headers={"X-API-KEY": config.get("api_key", "")},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                print(f"\n🔍 Found {len(results)} results for '{query}':\n")
                for r in results:
                    if isinstance(r, dict):
                        print(f"  • {r.get('text', r)}")
                    else:
                        print(f"  • {r}")
            else:
                print("No memories found for this query.")
        else:
            print(f"❌ Error: {resp.text}")
    except requests.ConnectionError:
        print("❌ Cannot connect to Memgraph server")


def status_cmd():
    """Check Memgraph connection status."""
    config = load_config()
    if not config:
        print("❌ Not initialized. Run 'memgraph init' first.")
        return

    print(f"📡 API URL: {config['api_url']}")
    print(f"🏢 Tenant:  {config['tenant_id']}")
    print(f"🔑 API Key: {'***' + config['api_key'][-4:] if config.get('api_key') else '(not set)'}")

    if requests:
        try:
            resp = requests.get(f"{config['api_url']}/health", timeout=5)
            if resp.status_code == 200:
                print("\n✅ Server is reachable")
            else:
                print(f"\n⚠️  Server returned: {resp.status_code}")
        except requests.ConnectionError:
            print("\n❌ Cannot connect to server")
    else:
        print("\n⚠️  Cannot check connection (requests not installed)")


def load_config():
    """Load config from .memgraph.env file."""
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        return None

    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                key = key.strip().lower().replace("memgraph_", "")
                config[key] = value.strip()

    return {
        "api_url": config.get("api_url", LOCAL_URL),
        "tenant_id": config.get("tenant_id"),
        "api_key": config.get("api_key")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Memgraph - AI Agent Memory Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  memgraph init                    # Set up Memgraph in current project
  memgraph remember "Used JWT"     # Store a memory
  memgraph recall "authentication" # Search memories
  memgraph status                  # Check connection
"""
    )

    subparsers = parser.add_subparsers(dest="command")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize Memgraph in current project")
    init_parser.add_argument("--non-interactive", action="store_true",
                             help="Non-interactive mode (reads from env vars: MEMGRAPH_API_URL, MEMGRAPH_TENANT_ID, MEMGRAPH_API_KEY)")

    # remember
    rem_parser = subparsers.add_parser("remember", help="Store a memory")
    rem_parser.add_argument("text", help="Text to remember")
    rem_parser.add_argument("-c", "--category", default="general",
                           choices=["decision", "architecture", "bug_fix", "preference", "general"])

    # recall
    rec_parser = subparsers.add_parser("recall", help="Search memories")
    rec_parser.add_argument("query", help="Search query")

    # status
    subparsers.add_parser("status", help="Check connection status")

    args = parser.parse_args()

    if args.command == "init":
        init_project(non_interactive=getattr(args, "non_interactive", False))
    elif args.command == "remember":
        remember_cmd(args.text, args.category)
    elif args.command == "recall":
        recall_cmd(args.query)
    elif args.command == "status":
        status_cmd()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
