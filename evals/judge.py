from __future__ import annotations

import asyncio
import os
import re
import weakref

from deepeval.models import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI

BASE_URL = os.getenv("AIROUTER_BASE_URL", "https://api.airouter.in/v1")
API_KEY = os.environ["AIROUTER_API_KEY"]
MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "openai/gpt-4o-mini")
TIMEOUT = float(os.getenv("DEEPEVAL_JUDGE_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("DEEPEVAL_JUDGE_RETRIES", "3"))
MAX_CONCURRENT_CALLS = int(os.getenv("DEEPEVAL_JUDGE_CONCURRENCY", "4"))

JSON_SUFFIX = (
    "\n\nRespond with ONLY a single valid JSON object. No markdown fences, "
    "no text outside the JSON."
)

SEMAPHORES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in SEMAPHORES:
        SEMAPHORES[loop] = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    return SEMAPHORES[loop]


class RouterJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = OpenAI(
            api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT, max_retries=MAX_RETRIES
        )
        self.async_client = AsyncOpenAI(
            api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT, max_retries=MAX_RETRIES
        )

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model

    def _request(self, prompt: str, schema) -> dict:
        if schema is not None:
            prompt += JSON_SUFFIX
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    @staticmethod
    def _parse(text: str, schema):
        if schema is None:
            return text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return schema.model_validate_json(m.group(0) if m else text)

    def generate(self, prompt: str, schema=None):
        res = self.client.chat.completions.create(**self._request(prompt, schema))
        return self._parse(res.choices[0].message.content, schema)

    async def a_generate(self, prompt: str, schema=None):
        async with get_semaphore():
            res = await self.async_client.chat.completions.create(**self._request(prompt, schema))
        return self._parse(res.choices[0].message.content, schema)


judge = RouterJudge()


if __name__ == "__main__":
    import time

    from pydantic import BaseModel

    class Answer(BaseModel):
        verdict: str

    t = time.time()
    print("model:", judge.get_model_name())
    print(judge.generate('Return {"verdict": "yes"}', schema=Answer))
    print(f"one call took {time.time() - t:.1f}s")