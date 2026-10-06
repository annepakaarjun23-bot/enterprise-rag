import { useState } from "react";
import { useChat } from "./hooks/useChat";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { ChatInput } from "./components/ChatInput";
import { StatusBadge } from "./components/StatusBadge";
import type { ChatOptions, RetrievalMode } from "./api/types";

const DEFAULT_OPTIONS: ChatOptions = {
  mode: "hybrid",
  topK: 5,
  automerge: true,
};

export default function App() {
  const { messages, status, error, sessionId, send, stop, reset } = useChat();
  const [options, setOptions] = useState<ChatOptions>(DEFAULT_OPTIONS);

  const busy = status === "retrieving" || status === "generating";

  return (
    <div className="h-full flex">
      <Sidebar onNewChat={reset} disabled={busy} />

      <main className="flex-1 flex flex-col min-w-0 bg-white">
        <header className="border-b border-neutral-200/70">
          <div className="px-6 py-3 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <h1 className="text-[13.5px] font-medium text-neutral-900 truncate">
                {messages.length === 0
                  ? "New conversation"
                  : messages.find((m) => m.role === "user")?.content ?? "Conversation"}
              </h1>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <select
                value={options.mode}
                onChange={(e) =>
                  setOptions((o) => ({
                    ...o,
                    mode: e.target.value as RetrievalMode,
                  }))
                }
                disabled={busy}
                className="text-[12px] border border-neutral-200 rounded-md px-2 py-1 bg-white text-neutral-700 hover:border-neutral-400 disabled:opacity-50 transition-colors"
              >
                <option value="hybrid">hybrid</option>
                <option value="dense">dense</option>
              </select>
            </div>
          </div>

          {busy && (
            <div className="px-6 pb-2">
              <StatusBadge status={status} />
            </div>
          )}
        </header>

        <ChatWindow messages={messages} onSuggestion={(t) => send(t, options)} />

        {error && (
          <div className="px-6">
            <div className="max-w-3xl mx-auto text-[12.5px] text-red-700 bg-red-50 border border-red-100 rounded-md px-3 py-2">
              {error}
            </div>
          </div>
        )}

        <ChatInput
          disabled={false}
          status={status}
          onSend={(t) => send(t, options)}
          onStop={stop}
        />

        {sessionId && (
          <div className="px-6 py-1.5 text-center">
            <span className="text-[10px] text-neutral-300 font-mono">
              {sessionId.slice(0, 8)}
            </span>
          </div>
        )}
      </main>
    </div>
  );
}