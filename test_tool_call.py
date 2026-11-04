"""Stage 2 smoke test: one Grok turn against the MCP server.

Starts mcp_server.py over stdio, hands Grok both tools, and prints every
tool call plus the final answer. No LangGraph.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# MCP stdio starts the server with fork+exec. gRPC's atfork handlers then log
# "skipping fork() handlers" because asyncio already has threads. The child
# does not use gRPC, so disable those handlers before importing the xAI SDK.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import grpc
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult
from xai_sdk import AsyncClient
from xai_sdk.chat import system, tool, tool_result, user

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "mcp_server.py"
MODEL = "grok-4.7"
USER_MESSAGE = "What's the current price and latest news for NVDA?"
MAX_TOOL_ROUNDS = 5
SYSTEM_PROMPT = (
    "You have tools for a live stock quote and recent headlines. "
    "Use them for any price or news the user asks about. "
    "Do not invent figures or headlines."
)


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into the environment without overriding existing vars."""
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def tool_payload(result: CallToolResult) -> str:
    """Serialize an MCP tool result for printing and for Grok's next message."""
    if result.structured_content is not None:
        body: object = result.structured_content
    else:
        body = [
            block.text if hasattr(block, "text") else str(block) for block in result.content
        ]
    return json.dumps(body, indent=2)


def parse_tool_args(raw: str) -> dict:
    """Parse a tool-call arguments JSON string into a dict."""
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Tool arguments must be a JSON object, got {type(parsed).__name__}")
    return parsed


async def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    load_env_file(ROOT / ".env")
    if not os.environ.get("XAI_API_KEY"):
        print("XAI_API_KEY is missing. Add it to .env and run this again.", file=sys.stderr)
        return 1

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        cwd=ROOT,
    )

    async with Client(server) as mcp_client:
        async with AsyncClient(api_key=os.environ["XAI_API_KEY"]) as xai:
            listed = await mcp_client.list_tools()
            tools = [
                tool(
                    name=mcp_tool.name,
                    description=mcp_tool.description or "",
                    parameters=mcp_tool.input_schema
                    or {"type": "object", "properties": {}},
                )
                for mcp_tool in listed.tools
            ]
            print(
                "Tools offered to Grok:",
                ", ".join(mcp_tool.name for mcp_tool in listed.tools),
            )
            print(f"\nUser: {USER_MESSAGE}\n")

            chat = xai.chat.create(
                model=MODEL,
                messages=[system(SYSTEM_PROMPT)],
                tools=tools,
            )
            chat.append(user(USER_MESSAGE))
            final_text = ""

            for _ in range(MAX_TOOL_ROUNDS + 1):
                try:
                    response = await chat.sample()
                except grpc.RpcError as exc:
                    details = exc.details() if callable(exc.details) else str(exc)
                    print(f"xAI API error ({exc.code()}): {details}", file=sys.stderr)
                    return 1

                chat.append(response)
                if not response.tool_calls:
                    final_text = (response.content or "").strip()
                    break

                for call in response.tool_calls:
                    args = parse_tool_args(call.function.arguments)
                    result = await mcp_client.call_tool(call.function.name, args)
                    returned = tool_payload(result)
                    print(f"Tool call: {call.function.name}")
                    print(f"Arguments: {json.dumps(args)}")
                    print(f"Returned:\n{returned}\n")
                    chat.append(tool_result(returned, tool_call_id=call.id))
            else:
                print(
                    f"Stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer.",
                    file=sys.stderr,
                )
                return 1

            print("Grok:")
            print(final_text or "(no text)")
            return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
