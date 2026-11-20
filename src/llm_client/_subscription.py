"""Shared base for clients that run a provider's local CLI on a subscription."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import sys
from abc import abstractmethod
from typing import ClassVar

from ._config import ModelConfig, ModelPricing
from .client import Client

# Subscription calls are not billed per token.
_SUBSCRIPTION_PRICING = ModelPricing(
    input_per_1m=0.0,
    output_per_1m=0.0,
    cached_input_per_1m=0.0,
    cache_creation_input_per_1m=0.0,
    cache_read_input_per_1m=0.0,
)


class SubscriptionClient(Client):
    """Runs each query as a local CLI process billed to the user's subscription.

    If the CLI is logged out, the first query starts its interactive login.
    Subclasses describe the CLI and implement `_auth_state` and `_query`.
    """

    cli: ClassVar[str]
    install_hint: ClassVar[str]
    login_args: ClassVar[tuple[str, ...]]
    subscription_name: ClassVar[str]
    # Environment variables that make the CLI bill an API key instead.
    api_key_env_vars: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        model_name: str | None = None,
        max_concurrency: int = 50,
        *,
        use_user_config: bool = False,
    ) -> None:
        # Skipping the user's CLI config avoids loading their MCP servers, plugins
        # and hooks on every call. Authentication is unaffected.
        self.use_user_config = use_user_config
        self._login_lock = asyncio.Lock()
        self._logged_in = False
        super().__init__(model_name=model_name, max_concurrency=max_concurrency)

    def _build_client(self) -> str:
        path = shutil.which(self.cli)
        if path is None:
            raise RuntimeError(
                f"The `{self.cli}` CLI was not found on PATH. {self.install_hint}"
            )
        return path

    def _billing_model(self) -> ModelConfig:
        # Tokens are tracked like the API client, but cost $0.
        return dataclasses.replace(self.model, pricing=_SUBSCRIPTION_PRICING)

    async def aclose(self) -> None:
        # Each query runs in its own process; there is nothing to close.
        return None

    @abstractmethod
    async def _auth_state(self) -> tuple[bool, str | None]:
        """Return whether the CLI is logged in, and why the login is unusable."""

    async def ensure_logged_in(self) -> None:
        """Run the CLI's interactive login if it is logged out."""
        if self._logged_in:
            return
        async with self._login_lock:
            if self._logged_in:
                return
            logged_in, problem = await self._auth_state()
            if not logged_in:
                print(
                    f"{self.cli} is not logged in. Starting `{self.cli} "
                    f"{' '.join(self.login_args)}`; sign in with your "
                    f"{self.subscription_name} account.",
                    file=sys.stderr,
                )
                # Inherit the terminal so the user sees the login URL and prompts.
                process = await asyncio.create_subprocess_exec(
                    self.client, *self.login_args, env=self._env()
                )
                if await process.wait() != 0:
                    raise RuntimeError(f"`{self.cli} login` failed")
                logged_in, problem = await self._auth_state()
                if not logged_in:
                    raise RuntimeError(f"{self.cli} is still not logged in")
            if problem:
                raise RuntimeError(problem)
            self._logged_in = True

    def _env(self) -> dict[str, str]:
        return {
            key: value
            for key, value in os.environ.items()
            if key not in self.api_key_env_vars
        }

    async def _run(
        self,
        *args: str,
        stdin: bytes | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        """Run the CLI, killing it on timeout or cancellation."""
        process = await asyncio.create_subprocess_exec(
            self.client,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=self._env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        return process.returncode or 0, stdout, stderr
