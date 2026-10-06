import json
import os
import random
from collections import defaultdict

from pydantic import BaseModel

from evals.judge import judge

random.seed(42)

N_SIMPLE = 30
N_MULTI = 10
MIN_TOKENS = 100

UNANSWERABLE = [
    "How do I deploy a LlamaIndex agent to AWS Lambda?",
    "What is the pricing of the OpenAI API?",
    "How do I train a custom embedding model with PyTorch?",
    "What is the capital of France?",
    "How do I configure Django middleware?",
]


class QA(BaseModel):
    input: str
    expected_output: str


PROMPT = """You are creating test questions for a RAG system over technical docs
(LangChain and LangGraph).

Based ONLY on the text below, write:
- "input": a question a developer would realistically ask. Do not copy phrases
  from the text and do not mention "the text" or "the document".
- "expected_output": a short, factual answer (1-2 sentences) found in the text.

{extra}

TEXT:
{text}

Return a JSON object with keys "input" and "expected_output"."""


def make_qa(text, extra=""):
    return judge.generate(PROMPT.format(text=text, extra=extra), schema=QA)


# 1. Load
with open("data/chunks/leaves.jsonl", encoding="utf-8") as f:
    all_chunks = [json.loads(line) for line in f if line.strip()]

by_id = {c["chunk_id"]: c for c in all_chunks}

# 2. Filter out short / tiny chunks
chunks = [
    c for c in all_chunks
    if not c["metadata"].get("short_leaf")
    and c["metadata"].get("token_count", 0) >= MIN_TOKENS
]
print(f"{len(chunks)} usable chunks out of {len(all_chunks)}")

# 3. Group by page, shuffle pages and chunks, then round-robin across pages
by_source = defaultdict(list)
for c in chunks:
    by_source[c["metadata"]["source_path"]].append(c)

sources = list(by_source)
random.shuffle(sources)
for items in by_source.values():
    random.shuffle(items)


def sample_spread(n):
    picked = []
    while len(picked) < n and any(by_source[s] for s in sources):
        for s in sources:
            if by_source[s] and len(picked) < n:
                picked.append(by_source[s].pop())
    return picked


def entry(dataset, qa, chunk_list, qtype):
    return {
        "id": f"q{len(dataset) + 1:03d}",
        "input": qa.input,
        "expected_output": qa.expected_output,
        "source_docs": [chunk_list[0]["metadata"]["source_path"]],
        "expected_chunk_ids": [c["chunk_id"] for c in chunk_list],
        "library": chunk_list[0]["metadata"]["library"],
        "type": qtype,
    }


dataset = []

# 4. Simple questions
for c in sample_spread(N_SIMPLE):
    qa = make_qa(c["text"])
    dataset.append(entry(dataset, qa, [c], "simple"))

# 5. Multi-chunk questions: a chunk + its next chunk (via next_chunk_id)
multi_count = 0
for c in sample_spread(N_MULTI * 4):  # extra candidates, some have no neighbor
    if multi_count >= N_MULTI:
        break
    nxt = by_id.get(c.get("next_chunk_id"))
    if not nxt or nxt["metadata"]["source_path"] != c["metadata"]["source_path"]:
        continue
    qa = make_qa(
        c["text"] + "\n\n---\n\n" + nxt["text"],
        extra="The question must require information from BOTH parts of the text.",
    )
    dataset.append(entry(dataset, qa, [c, nxt], "multi_chunk"))
    multi_count += 1

# 6. Unanswerable questions
for q in UNANSWERABLE:
    dataset.append({
        "id": f"q{len(dataset) + 1:03d}",
        "input": q,
        "expected_output": "I don't have information about that in the provided documents.",
        "source_docs": [],
        "expected_chunk_ids": [],
        "library": None,
        "type": "unanswerable",
    })

os.makedirs("evals/datasets", exist_ok=True)
with open("evals/datasets/golden_retriever.jsonl", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Saved {len(dataset)} questions")