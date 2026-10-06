from __future__ import annotations

import asyncio
import sys
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def setup_windows_event_loop() -> None:

    if sys.platform != "win32":
        return
    policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_cls is not None:
        asyncio.set_event_loop_policy(policy_cls())


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    if sys.platform == "win32":
        return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(coro)


def uvicorn_loop_arg() -> str:
    if sys.platform != "win32":
        return "auto"
    try:
        from importlib.metadata import version
        parts = version("uvicorn").split(".")
        major, minor = int(parts[0]), int(parts[1])
        if (major, minor) >= (0, 36):
            return "asyncio:SelectorEventLoop"
        return "none"
    except Exception:
        return "asyncio:SelectorEventLoop"