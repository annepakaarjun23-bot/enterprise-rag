from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.api.middleware import RequestIDMiddleware  
from rag.api.routes.chat import router as chat_router  
from rag.api.routes.health import router as health_router  
from rag.graph.checkpointer import get_checkpointer  
from rag.graph.graph_build import build_rag_graph  
from rag.retrieval.retriever import RetrieverConfig, get_retriever  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()

    logger.info("startup: opening Postgres checkpointer pool")
    checkpointer, pool = await get_checkpointer()
    app.state.pool = pool

    logger.info("startup: compiling LangGraph")
    app.state.graph = build_rag_graph(checkpointer)

    logger.info("startup: warming retriever cache (default config)")
    try:
        get_retriever(RetrieverConfig())
    except Exception as e: 
        logger.warning("retriever warm failed: %s", e)

    app.state.started_at = time.time()
    logger.info("startup complete in %.0fms", (time.perf_counter() - t0) * 1000)

    try:
        yield
    finally:
        logger.info("shutdown: closing checkpointer pool")
        await pool.close()


app = FastAPI(
    title="Enterprise RAG API",
    description="RAG over LangChain + LangGraph documentation with hybrid retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)
app.include_router(chat_router)
app.include_router(health_router)