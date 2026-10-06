from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag._compat import setup_windows_event_loop, uvicorn_loop_arg  
setup_windows_event_loop()

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "rag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        loop=uvicorn_loop_arg(),
    )