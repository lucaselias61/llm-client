"""Claude Code CLI client: runs Anthropic models with `claude -p` on a Claude plan."""

from __future__ import annotations

import json
import tempfile
from typing import Any

from pydantic import BaseModel

from ._subscription import SubscriptionClient

# Without a system prompt Claude Code presents itself as a coding agent.
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class ClaudeClient(SubscriptionClient):
    """Runs Anthropic models through the local `claude` CLI on your Claude plan.

    Selected by `LLMClient(model_name, use_subscription=True)` for Anthropic
    models. Each query runs `claude -p --output-format json --json-schema ...`
    with no tools in an empty temporary directory.
    """

    provider = "anthropic"
    cli = "claude"
    install_hint = "Install it with `npm install -g @anthropic-ai/claude-code`."
    login_args = ("auth", "login", "--claudeai")
    subscription_name = "Claude"
    # Claude Code prefers these over the subscription login when they are set.
    api_key_env_vars = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

    async def _auth_state(self) -> tuple[bool, str | None]:
        returncode, stdout, _ = await self._run("auth", "status", "--json")
        try:
            status = json.loads(stdout)
        except json.JSONDecodeError:
            status = {}
        if returncode != 0 or not status.get("loggedIn"):
            return False, None
        method = status.get("authMethod")
        api_provider = status.get("apiProvider", "firstParty")
        if method != "claude.ai" or api_provider != "firstParty":
            return True, (
                f"Claude Code is logged in with '{method}' via '{api_provider}', "
                "which does not use your Claude subscription. Run "
                "`claude auth logout` and then `claude auth login --claudeai`."
            )
        return True, None

    async def _query(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        text_format: type[BaseModel] | None = None,
        reasoning_effort: str | None = None,
    ) -> Any:
        # `temperature` is accepted for the shared interface; Claude Code does not
        # expose it.
        await self.ensure_logged_in()

        args = [
            "-p",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--model",
            self.model.model_name,
            "--tools",
            "",
            "--disable-slash-commands",
            "--system-prompt",
            system_prompt or _DEFAULT_SYSTEM_PROMPT,
        ]
        if not self.use_user_config:
            args += ["--setting-sources", "", "--strict-mcp-config"]
        if reasoning_effort:
            args += ["--effort", reasoning_effort]
        if text_format is not None:
            args += ["--json-schema", json.dumps(text_format.model_json_schema())]

        # The empty working directory keeps CLAUDE.md files out of the context.
        with tempfile.TemporaryDirectory(prefix="llm-client-claude-") as tmp:
            # Read the prompt from stdin so long prompts are not limited by argv size.
            returncode, stdout, stderr = await self._run(
                *args, stdin=prompt.encode(), cwd=tmp, timeout=timeout
            )

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            result = None
        if not isinstance(result, dict) or returncode != 0 or result.get("is_error"):
            message = (result.get("result") if isinstance(result, dict) else None) or (
                stdout + stderr
            ).decode(errors="replace").strip()
            raise RuntimeError(f"claude -p failed: {message}")

        usage = result.get("usage") or {}
        self._add_response_usage(
            input_tokens=usage.get("input_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=(usage.get("output_tokens_details") or {}).get(
                "thinking_tokens", 0
            ),
        )

        if text_format is not None:
            # The base class validates this against text_format.
            return result.get("structured_output")
        return result.get("result")
