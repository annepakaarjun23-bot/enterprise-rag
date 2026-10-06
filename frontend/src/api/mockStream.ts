import type { ChatChunk, ChatOptions, SSEEvent } from "./types";

/**
 * Demo-mode stream. Emits the exact same SSE event sequence as the FastAPI
 * backend so the frontend can't tell the difference. Ships in the deployed
 * build when VITE_DEMO_MODE is not "false".
 *
 * Each canned response has keyword triggers and plausible sources drawn from
 * the same corpus shapes the real backend returns (library/version, heading
 * path, chunk text). Falls back to a general response when nothing matches.
 */

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

function chunkText(text: string): string[] {
  // split into word-ish tokens, keeping trailing whitespace, so joined
  // output equals the original exactly
  return text.match(/\S+\s*|\s+/g) ?? [text];
}

function randomSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2, 12);
}

// ---------------------------------------------------------------------------
// canned responses
// ---------------------------------------------------------------------------

interface MockResponse {
  keywords: string[];
  answer: string;
  sources: Omit<ChatChunk, "chunk_id" | "parent_chunk_id">[];
  fallback?: boolean;
}

const RESPONSES: MockResponse[] = [
  // ----------------------------------------------------------------
  // memory / checkpointers
  // ----------------------------------------------------------------
  {
    keywords: [
      "memory",
      "remember",
      "checkpointer",
      "thread",
      "persist",
      "short-term",
      "long-term",
    ],
    answer: `In LangGraph, memory is managed through **checkpointers**. A checkpointer persists graph state after each step, so the conversation can be resumed by passing the same \`thread_id\` in the config.

For short-term (thread-scoped) memory, compile the graph with a checkpointer:

\`\`\`python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState

graph = (
    StateGraph(MessagesState)
    .add_node("agent", call_model)
    .add_edge("__start__", "agent")
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)
\`\`\`

Use \`InMemorySaver\` for development and \`PostgresSaver\` or \`SqliteSaver\` for production, since in-memory state is lost when the process restarts.

For **long-term** (cross-thread) memory, use a \`BaseStore\` — \`InMemoryStore\` locally, \`PostgresStore\` in production — and access it inside your nodes via the runtime object.`,
    sources: [
      {
        text: "A checkpointer persists the graph state at every step, keyed by thread_id. Compile the graph with .compile(checkpointer=...) and pass the thread_id inside the config when invoking. Without a checkpointer, each invocation starts fresh and no memory is retained between turns.",
        score: 0.83,
        library: "langgraph",
        version: "1.x",
        source_path: "langgraph/persistence.mdx",
        source_url: "",
        heading_path: "Persistence > Checkpointers",
        has_code: true,
      },
      {
        text: "## Short-term memory\n\nShort-term memory is thread-scoped: it holds the conversation history for one session. LangGraph persists this via checkpoints. Use InMemorySaver for development, and PostgresSaver or SqliteSaver for production workloads where state must survive a process restart.",
        score: 0.79,
        library: "langgraph",
        version: "1.x",
        source_path: "langgraph/add-memory.mdx",
        source_url: "",
        heading_path: "Add memory > Add short-term memory",
        has_code: true,
      },
      {
        text: "Long-term memory is stored outside the thread, using a BaseStore implementation. Access it from inside a node via runtime.store. Namespaces let you scope entries by user, application, or any custom key.",
        score: 0.71,
        library: "langgraph",
        version: "1.x",
        source_path: "langgraph/add-memory.mdx",
        source_url: "",
        heading_path: "Add memory > Add long-term memory",
        has_code: false,
      },
    ],
  },

  // ----------------------------------------------------------------
  // create_agent
  // ----------------------------------------------------------------
  {
    keywords: [
      "create_agent",
      "create an agent",
      "build an agent",
      "make an agent",
      "what is create",
    ],
    answer: `\`create_agent\` is the high-level entry point for building an agent in LangChain 1.x. It returns a compiled LangGraph graph you can invoke or stream like any other runnable.

\`\`\`python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny in {city}"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]}
)
\`\`\`

Under the hood, \`create_agent\` builds a \`StateGraph\` with a model node and a tools node, wired with a conditional edge that loops between them until the model stops requesting tool calls.

For more control — custom state, custom routing, human-in-the-loop — build the graph directly with \`StateGraph\` from LangGraph.`,
    sources: [
      {
        text: "## create_agent\n\ncreate_agent is the standard way to construct a tool-calling agent in LangChain 1.x. It accepts a model, a list of tools, and an optional system prompt, and returns a compiled graph. The graph loops between a model node and a tools node until the model produces a response with no tool calls.",
        score: 0.86,
        library: "langchain",
        version: "1.x",
        source_path: "langchain/agents.mdx",
        source_url: "",
        heading_path: "Agents > create_agent",
        has_code: false,
      },
      {
        text: "create_agent is built on top of LangGraph's StateGraph. If you need to customize the graph — adding middleware, extra nodes, or interrupts — you can either pass middleware to create_agent or build the graph directly with StateGraph.",
        score: 0.74,
        library: "langchain",
        version: "1.x",
        source_path: "langchain/agents.mdx",
        source_url: "",
        heading_path: "Agents > create_agent > Built on LangGraph",
        has_code: false,
      },
    ],
  },

  // ----------------------------------------------------------------
  // streaming
  // ----------------------------------------------------------------
  {
    keywords: ["stream", "streaming", "tokens", "sse", "server-sent", "chunk"],
    answer: `LangGraph streams through \`graph.stream()\` (sync) or \`graph.astream()\` (async). The \`stream_mode\` argument controls what gets emitted:

| mode | emits |
|---|---|
| \`"values"\` | full state after each step |
| \`"updates"\` | just the node's output |
| \`"messages"\` | LLM tokens as they're produced |
| \`"custom"\` | anything you push via \`get_stream_writer()\` |

For token-by-token output from an agent, use \`stream_mode="messages"\`:

\`\`\`python
async for chunk in graph.astream(
    {"messages": [{"role": "user", "content": "hi"}]},
    stream_mode="messages",
):
    msg, meta = chunk
    if msg.content:
        print(msg.content, end="", flush=True)
\`\`\`

To emit your own events — retrieval stages, custom progress — call \`get_stream_writer()\` inside a node and yield plain dicts. Those arrive under \`stream_mode="custom"\`.`,
    sources: [
      {
        text: "Stream modes control what the graph emits. 'values' yields the full state after each step, 'updates' yields only the changes from each node, 'messages' yields LLM tokens with metadata, and 'custom' yields anything written by get_stream_writer(). Multiple modes can be combined into one stream.",
        score: 0.88,
        library: "langgraph",
        version: "1.x",
        source_path: "langgraph/streaming.mdx",
        source_url: "",
        heading_path: "Streaming > Stream modes",
        has_code: false,
      },
      {
        text: "To stream custom data from inside a node, import get_stream_writer from langgraph.config, call it to get a writer function, and pass any JSON-serializable payload. Consume those events with stream_mode='custom'.",
        score: 0.76,
        library: "langgraph",
        version: "1.x",
        source_path: "langgraph/streaming.mdx",
        source_url: "",
        heading_path: "Streaming > Stream custom data",
        has_code: true,
      },
    ],
  },

  // ----------------------------------------------------------------
  // migration 0.6 -> 1.x
  // ----------------------------------------------------------------
  {
    keywords: ["migrat", "0.6", "upgrade", "breaking change", "what changed"],
    answer: `The main breaking changes between LangGraph 0.6 and 1.x, and between LangChain 0.3 and 1.x:

- \`create_react_agent\` was replaced by \`create_agent\` from \`langchain.agents\`
- The default streaming node name changed from \`"agent"\` to \`"model"\` to better reflect the node's role
- \`convert_mcp_tool_to_langchain_tool\` was renamed to \`as_langchain_tool\`, and it is now a coroutine
- Prompted output via \`response_format\` was removed because it wasn't reliable compared to native structured output

Migration is mostly a find-and-replace on imports, but check your streaming handlers if you filter by node name:

\`\`\`python
# before (0.6)
if node_name == "agent":
    ...

# after (1.x)
if node_name == "model":
    ...
\`\`\`

The deprecation shims were removed in 1.0, so the old names raise immediately rather than warning.`,
    sources: [
      {
        text: "## Migrate to create_agent\n\ncreate_react_agent has been replaced by create_agent, which lives in langchain.agents. The new API returns a compiled graph and accepts a model, tools list, and system prompt. The old name continues to work with a deprecation warning in 0.3, and raises in 1.0.",
        score: 0.84,
        library: "langchain",
        version: "1.x",
        source_path: "langchain/migrate.mdx",
        source_url: "",
        heading_path: "Migrate > Import path",
        has_code: true,
      },
      {
        text: "When streaming events from agents, the node name changed from 'agent' to 'model' to better reflect the node's purpose. Any code that filters stream events by node name needs updating.",
        score: 0.78,
        library: "langgraph",
        version: "1.x",
        source_path: "langgraph/migrate.mdx",
        source_url: "",
        heading_path: "Migrate > Streaming changes",
        has_code: false,
      },
    ],
  },

  // ----------------------------------------------------------------
  // tool errors
  // ----------------------------------------------------------------
  {
    keywords: [
      "tool error",
      "toolmessage",
      "tool call error",
      "handle error",
      "retry",
      "exception in tool",
    ],
    answer: `The \`ToolMessage content must be a string\` error happens when a tool returns something other than a string — a dict, a Pydantic model, or a list. LangChain wraps tool outputs in a \`ToolMessage\`, and that wrapper requires a string payload.

Two ways to fix it:

**Return a string:**

\`\`\`python
from langchain.tools import tool

@tool
def lookup(key: str) -> str:
    """Look up a value."""
    return json.dumps({"key": key, "value": 42})
\`\`\`

**Or build the ToolMessage yourself:**

\`\`\`python
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime

@tool
def lookup(key: str, runtime: ToolRuntime) -> ToolMessage:
    """Look up a value."""
    return ToolMessage(
        content=json.dumps({"key": key, "value": 42}),
        tool_call_id=runtime.tool_call_id,
    )
\`\`\`

For retryable failures — network timeouts, rate limits — raise inside the tool and let the agent's middleware retry policy handle it, rather than catching and returning an error string.`,
    sources: [
      {
        text: "Tools must return strings, or a ToolMessage with string content. If your tool returns a dict or a Pydantic model, wrap it with json.dumps() or construct a ToolMessage directly.",
        score: 0.81,
        library: "langchain",
        version: "1.x",
        source_path: "langchain/tools.mdx",
        source_url: "",
        heading_path: "Tools > Return values",
        has_code: true,
      },
      {
        text: "To construct a ToolMessage inside a tool, access runtime.tool_call_id and pass it as the tool_call_id parameter. This keeps the message properly linked to the originating tool call.",
        score: 0.74,
        library: "langchain",
        version: "1.x",
        source_path: "langchain/tools.mdx",
        source_url: "",
        heading_path: "Tools > ToolMessage",
        has_code: true,
      },
    ],
  },

  // ----------------------------------------------------------------
  // fallback
  // ----------------------------------------------------------------
  {
    fallback: true,
    keywords: [],
    answer: `Here's what I know from the retrieved documentation.

The concept you're asking about touches a few areas of the LangChain and LangGraph APIs. In the current docs, the general approach is:

1. **Define the state** you want to track across the graph. For agent-style flows, this usually includes a \`messages\` list annotated with \`add_messages\`.

2. **Add nodes** that read from and write to that state — a model node for LLM calls, a tools node for tool dispatch, any custom logic nodes.

3. **Wire the edges**. Use \`add_edge\` for unconditional transitions and \`add_conditional_edges\` for branching. Compile with a checkpointer if you need persistence.

\`\`\`python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("tools", call_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "model")
graph = builder.compile()
\`\`\`

For a more specific answer, try asking about a particular API, class, or error you're seeing — those map to concrete pages in the docs.`,
    sources: [
      {
        text: "StateGraph is the core building block in LangGraph. Define a state schema, add nodes that read and write to it, and wire edges between them. Compile the graph to get a runnable that you can invoke, stream, or persist with a checkpointer.",
        score: 0.62,
        library: "langgraph",
        version: "1.x",
        source_path: "langgraph/graph-api.mdx",
        source_url: "",
        heading_path: "Graph API > Overview",
        has_code: true,
      },
      {
        text: "Conditional edges route execution based on a function of the state. The function returns a string key that maps to the next node name, or to END to stop the graph.",
        score: 0.58,
        library: "langgraph",
        version: "1.x",
        source_path: "langgraph/graph-api.mdx",
        source_url: "",
        heading_path: "Graph API > Conditional edges",
        has_code: false,
      },
    ],
  },
];

function pickResponse(message: string): MockResponse {
  const lower = message.toLowerCase();
  let best: MockResponse | null = null;
  let bestScore = 0;

  for (const r of RESPONSES) {
    if (r.fallback) continue;
    let score = 0;
    for (const kw of r.keywords) {
      if (lower.includes(kw)) score += 1;
    }
    if (score > bestScore) {
      best = r;
      bestScore = score;
    }
  }

  return best ?? RESPONSES.find((r) => r.fallback)!;
}

// ---------------------------------------------------------------------------
// stream
// ---------------------------------------------------------------------------

export async function* streamMock(
  message: string,
  sessionId: string | null,
  options: ChatOptions,
  signal: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const t0 = performance.now();
  const sid = sessionId ?? randomSessionId();

  const response = pickResponse(message);

  // session
  yield { type: "session", session_id: sid };
  await sleep(120, signal);

  // retrieval stage
  yield {
    type: "stage",
    node: "retriever",
    message: `Retrieving (${options.mode}, top_k=${options.topK}) for: ${message}`,
  };
  await sleep(420, signal);

  // sources arrive one at a time
  const chunks: ChatChunk[] = response.sources.map((s, i) => ({
    ...s,
    chunk_id: `demo-${sid.slice(0, 6)}-${i}`,
    parent_chunk_id: `demo-parent-${sid.slice(0, 6)}-${i}`,
  }));
  for (const chunk of chunks) {
    yield { type: "chunk", chunk };
    await sleep(140, signal);
  }

  yield { type: "stage_done", node: "retriever" };
  await sleep(180, signal);

  // generation stage
  yield {
    type: "stage",
    node: "generator",
    message: "Generating answer...",
  };
  await sleep(260, signal);

  // tokens
  const tokens = chunkText(response.answer);
  for (const text of tokens) {
    yield { type: "token", text };
    // small jitter so it feels like a real model
    await sleep(12 + Math.random() * 22, signal);
  }

  // done
  yield {
    type: "done",
    session_id: sid,
    answer: response.answer,
    chunks,
    duration_ms: Math.round(performance.now() - t0),
  };
}