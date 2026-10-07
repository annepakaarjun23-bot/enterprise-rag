import type { ChatOptions, SSEEvent } from "./types";
import { streamMock } from "./mockStream";

/**
 * Demo mode is ON by default so the deployed build works with no backend.
 * Set VITE_DEMO_MODE=false in .env.local to hit the real FastAPI backend.
 */
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE !== "false";

/**
 * Stream chat events.
 *
 * In demo mode, yields canned responses with realistic timing.
 * Otherwise POSTs to /chat/stream and parses SSE frames from the response
 * body (EventSource only speaks GET, so we read the stream ourselves).
 */
export async function* streamChat(
  message: string,
  sessionId: string | null,
  options: ChatOptions,
  signal: AbortSignal,
): AsyncGenerator<SSEEvent> {
  if (DEMO_MODE) {
    yield* streamMock(message, sessionId, options, signal);
    return;
  }

  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      retrieval_mode: options.mode,
      top_k: options.topK,
      use_automerge: options.automerge,
    }),
    signal,
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Request failed (${response.status}): ${body.slice(0, 200)}`);
  }
  if (!response.body) {
    throw new Error("No response body — streaming not supported by this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const dataLine = frame
          .split("\n")
          .find((line) => line.startsWith("data:"));
        if (!dataLine) continue;

        const json = dataLine.slice("data:".length).trim();
        if (!json) continue;

        try {
          yield JSON.parse(json) as SSEEvent;
        } catch {
          console.warn("skipping malformed SSE frame:", json);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}