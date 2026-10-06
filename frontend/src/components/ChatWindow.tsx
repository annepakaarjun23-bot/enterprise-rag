import { useEffect, useRef } from "react";
import type { Message as MessageType } from "../api/types";
import { Message } from "./Message";

interface ChatWindowProps {
  messages: MessageType[];
  onSuggestion: (text: string) => void;
}

const SUGGESTIONS = [
  "How do I add memory to a LangGraph agent?",
  "What is create_agent?",
  "How do I stream tokens from a LangGraph agent?",
  "What changed between LangGraph 0.6 and 1.x?",
  "How do I handle tool errors in an agent?",
];

export function ChatWindow({ messages, onSuggestion }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-16">
          <h1 className="text-2xl font-semibold text-neutral-900">
            LangChain + LangGraph docs, answered.
          </h1>
          <p className="mt-2 text-sm text-neutral-500 leading-relaxed">
            Answers are grounded in the official documentation and cite the
            sections they were drawn from. Hybrid retrieval (dense + BM25,
            fused with reciprocal rank fusion), parent-child auto-merging,
            streaming responses.
          </p>

          <div className="mt-8 space-y-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => onSuggestion(s)}
                className="block w-full text-left px-3 py-2.5 rounded-lg border border-neutral-200 bg-white hover:border-neutral-400 hover:bg-neutral-50 text-sm text-neutral-700 transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-4 py-6 space-y-5">
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}