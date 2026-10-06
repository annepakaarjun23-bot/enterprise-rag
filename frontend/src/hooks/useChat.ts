import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat } from "../api/stream";
import type {
  ChatChunk,
  ChatOptions,
  ChatStatus,
  Message,
  SSEEvent,
} from "../api/types";

const SESSION_KEY = "rag_session_id";

// Smooth-streaming tunables.
const TICK_MS = 24;              // drain cadence
const BASE_CHARS = 3;            // characters per tick at rest
const FAST_CHARS = 9;            // characters per tick when backlog is large
const BACKLOG_THRESHOLD = 80;    // chars above which we speed up

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(SESSION_KEY);
    } catch {
      return null;
    }
  });

  const abortRef = useRef<AbortController | null>(null);

  // Streaming buffer:
  //   bufferRef.text  — received tokens not yet shown
  //   bufferRef.id    — which message they belong to
  //   bufferRef.done  — set when the 'done' event arrived; finalize when drained
  const bufferRef = useRef<{ id: string | null; text: string; done: null | {
    answer: string;
    chunks: ChatChunk[];
    durationMs: number;
  } }>({ id: null, text: "", done: null });

  // Drain ticker — appends a small slice of the buffer to the message on
  // each tick. This is what makes the stream feel like typing instead of a
  // wall of text appearing at once.
  useEffect(() => {
    const timer = window.setInterval(() => {
      const buf = bufferRef.current;
      if (!buf.id) return;

      if (buf.text.length > 0) {
        const speed = buf.text.length > BACKLOG_THRESHOLD ? FAST_CHARS : BASE_CHARS;
        const slice = buf.text.slice(0, speed);
        buf.text = buf.text.slice(speed);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === buf.id ? { ...m, content: m.content + slice } : m,
          ),
        );
        return;
      }

      // buffer empty and 'done' arrived -> finalize the message
      if (buf.done) {
        const fin = buf.done;
        const id = buf.id;
        buf.id = null;
        buf.done = null;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === id
              ? {
                  ...m,
                  content: m.content.length ? m.content : fin.answer,
                  chunks: fin.chunks.length ? fin.chunks : m.chunks,
                  streaming: false,
                  durationMs: fin.durationMs,
                }
              : m,
          ),
        );
        setStatus("idle");
      }
    }, TICK_MS);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    try {
      if (sessionId) localStorage.setItem(SESSION_KEY, sessionId);
    } catch {
      /* ignore */
    }
  }, [sessionId]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    bufferRef.current = { id: null, text: "", done: null };
    setMessages([]);
    setStatus("idle");
    setError(null);
    setSessionId(null);
    try {
      localStorage.removeItem(SESSION_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    // flush whatever is in the buffer immediately
    const buf = bufferRef.current;
    if (buf.id) {
      const id = buf.id;
      const remainder = buf.text;
      buf.id = null;
      buf.text = "";
      buf.done = null;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                content: m.content + remainder,
                streaming: false,
                error: "Stopped by user",
              }
            : m,
        ),
      );
    }
    setStatus("idle");
  }, []);

  const send = useCallback(
    async (text: string, options: ChatOptions) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      if (status === "retrieving" || status === "generating") return;

      setError(null);
      setStatus("retrieving");

      const userMsg: Message = { id: newId(), role: "user", content: trimmed };
      const assistantId = newId();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        chunks: [],
        streaming: true,
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      bufferRef.current = { id: assistantId, text: "", done: null };

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        for await (const event of streamChat(
          trimmed,
          sessionId,
          options,
          controller.signal,
        )) {
          handleEvent(event);
        }
      } catch (err) {
        const name = (err as Error)?.name;
        if (name === "AbortError") {
          setStatus("idle");
        } else {
          const message = (err as Error)?.message ?? "Unknown error";
          setError(message);
          setStatus("error");
          bufferRef.current.id = null;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, streaming: false, error: message }
                : m,
            ),
          );
        }
      } finally {
        abortRef.current = null;
      }
    },
    [sessionId, status],
  );

  function handleEvent(event: SSEEvent) {
    switch (event.type) {
      case "session":
        if (event.session_id !== sessionId) setSessionId(event.session_id);
        break;

      case "stage":
        setStatus(event.node === "retriever" ? "retrieving" : "generating");
        break;

      case "stage_done":
        break;

      case "chunk":
        setMessages((prev) =>
          prev.map((m) =>
            m.id === bufferRef.current.id
              ? { ...m, chunks: [...(m.chunks ?? []), event.chunk] }
              : m,
          ),
        );
        break;

      case "token":
        bufferRef.current.text += event.text;
        break;

      case "error":
        setError(event.message);
        bufferRef.current.id = null;
        setMessages((prev) =>
          prev.map((m) =>
            m.streaming ? { ...m, streaming: false, error: event.message } : m,
          ),
        );
        setStatus("error");
        break;

      case "done":
        bufferRef.current.done = {
          answer: event.answer,
          chunks: event.chunks ?? [],
          durationMs: event.duration_ms,
        };
        break;
    }
  }

  return { messages, status, error, sessionId, send, stop, reset };
}