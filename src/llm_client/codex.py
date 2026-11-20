"""Codex CLI client: runs OpenAI models with `codex exec` on a ChatGPT plan."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from ._subscription import SubscriptionClient


class CodexClient(SubscriptionClient):
    """Runs OpenAI models through the local `codex` CLI on your ChatGPT plan.

    Selected by `LLMClient(model_name, use_subscription=True)` for OpenAI models.
    Each query runs `codex exec --output-schema schema.json -o out.json` in an
    empty temporary directory with a read-only sandbox.
    """

    provider = "openai"
    cli = "codex"
    install_hint = (
        "Install it with `npm install -g @openai/codex` or `brew install codex`."
    )
    login_args = ("login",)
    subscription_name = "ChatGPT"

    async def _auth_state(self) -> tuple[bool, str | None]:
        returncode, stdout, stderr = await self._run("login", "status")
        status = (stdout + stderr).decode(errors="replace").strip()
        if returncode != 0 or "not logged in" in status.lower():
            return False, None
        if "chatgpt" not in status.lower():
            return True, (
                f"Codex reports '{status}', which bills API tokens instead of your "
                "ChatGPT subscription. Run `codex logout` and then `codex login` "
                "to sign in with ChatGPT."
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
        # `temperature` is accepted for the shared interface; Codex does not expose it.
        await self.ensure_logged_in()

        with tempfile.TemporaryDirectory(prefix="llm-client-codex-") as tmp:
            workdir = Path(tmp)
            output_path = workdir / "out.json"
            args = [
                "exec",
                "--json",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                str(workdir),
                "--model",
                self.model.model_name,
                "--output-last-message",
                str(output_path),
            ]
            if not self.use_user_config:
                args.append("--ignore-user-config")
            if system_prompt:
                args += ["-c", f"developer_instructions={json.dumps(system_prompt)}"]
            if reasoning_effort:
                args += ["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"]
            if text_format is not None:
                schema_path = workdir / "schema.json"
                schema_path.write_text(json.dumps(to_strict_json_schema(text_format)))
                args += ["--output-schema", str(schema_path)]
            # Read the prompt from stdin so long prompts are not limited by argv size.
            args.append("-")

            returncode, stdout, stderr = await self._run(
                *args, stdin=prompt.encode(), timeout=timeout
            )
            events = _parse_events(stdout)
            if returncode != 0:
                raise RuntimeError(
                    "codex exec failed: "
                    + (
                        _error_message(events)
                        or stderr.decode(errors="replace").strip()
                    )
                )

            usage = (
                next(
                    (
                        e.get("usage")
                        for e in reversed(events)
                        if e.get("type") == "turn.completed"
                    ),
                    None,
                )
                or {}
            )
            self._add_response_usage(
                input_tokens=usage.get("input_tokens", 0),
                cached_tokens=usage.get("cached_input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                reasoning_tokens=usage.get("reasoning_output_tokens", 0),
            )

            if not output_path.exists():
                return None
            # With a schema this is a JSON string that the base class validates.
            return output_path.read_text()


def _parse_events(stdout: bytes) -> list[dict[str, Any]]:
    events = []
    for line in stdout.decode(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _error_message(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("type") == "turn.failed":
            return (event.get("error") or {}).get("message")
        if event.get("type") == "error":
            return event.get("message")
    return None
