import { useEffect, useRef, useState } from "react";
import type { ChatStatus } from "../api/types";

interface ChatInputProps {
  disabled: boolean;
  status: ChatStatus;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function ChatInput({ disabled, status, onSend, onStop }: ChatInputProps) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const busy = status === "retrieving" || status === "generating";

  const submit = () => {
    const text = value.trim();
    if (!text || disabled || busy) return;
    onSend(text);
    setValue("");
  };

  return (
    <div className="border-t border-neutral-200/70 bg-white">
      <div className="max-w-3xl mx-auto px-6 py-4">
        <div className="flex items-end gap-2 border border-neutral-300 rounded-2xl px-3 py-2 bg-white focus-within:border-neutral-500 focus-within:ring-2 focus-within:ring-neutral-900/5 transition-all">
          <textarea
            ref={ref}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask about LangChain or LangGraph…"
            rows={1}
            disabled={disabled}
            className="flex-1 resize-none bg-transparent outline-none text-[15px] leading-relaxed placeholder:text-neutral-400 disabled:opacity-50"
          />
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              className="shrink-0 size-8 rounded-full grid place-items-center bg-neutral-900 text-white hover:bg-neutral-700 transition-colors"
              title="Stop"
            >
              <span className="block size-3 bg-white rounded-[2px]" />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={disabled || !value.trim()}
              className="shrink-0 size-8 rounded-full grid place-items-center bg-neutral-900 text-white hover:bg-neutral-700 disabled:bg-neutral-200 disabled:text-neutral-400 disabled:cursor-not-allowed transition-colors"
              title="Send"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          )}
        </div>
        <div className="mt-2 text-[11px] text-neutral-400 text-center">
          Enter to send · Shift+Enter for newline
        </div>
      </div>
    </div>
  );
}