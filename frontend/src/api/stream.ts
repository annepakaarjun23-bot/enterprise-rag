import type { ChatOptions, SSEEvent } from "./types";
import { streamMock } from "./mockStream";

/**
 * Demo mode is ON by default so the deployed build works with no backend.
 * Set VITE_DEMO_MODE=false in .env.local to hit the real FastAPI backend
 * during development.
 */
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE !== "false";

/**
 * Stream chat events.
 *
 * In demo mode, returns a canned response streamed with realistic timing.
 * Otherwise posts to /chat/stream and parses SSE frames from the response
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

  // ---- real backend path (unchanged) ----
  const response = await fetch("/chat/stream", {
    // ... rest of the existing implementation ...
  });
  // ...
}