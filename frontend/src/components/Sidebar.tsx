interface SidebarProps {
  onNewChat: () => void;
  disabled?: boolean;
}

interface StackRow {
  label: string;
  value: string;
}

const STACK: StackRow[] = [
  { label: "Retrieval", value: "hybrid · dense + BM25" },
  { label: "Fusion", value: "reciprocal rank" },
  { label: "Reranking", value: "off" },
  { label: "AutoMerge", value: "on" },
  { label: "Top-k", value: "5" },
  { label: "Docs", value: "150 · 3 versions" },
];

export function Sidebar({ onNewChat, disabled }: SidebarProps) {
  return (
    <aside className="w-64 shrink-0 bg-neutral-50 border-r border-neutral-200 flex flex-col">
      {/* brand */}
      <div className="px-4 py-4 flex items-center gap-2.5 border-b border-neutral-200">
        <div className="size-7 rounded-md bg-neutral-900 grid place-items-center shrink-0">
          <span className="text-white text-xs font-bold">R</span>
        </div>
        <div className="min-w-0">
          <div className="text-[13px] font-semibold text-neutral-900 leading-tight">
            Enterprise RAG
          </div>
          <div className="text-[10.5px] text-neutral-500 leading-tight truncate">
            LangChain + LangGraph
          </div>
        </div>
      </div>

      {/* new chat */}
      <div className="px-3 py-3">
        <button
          type="button"
          onClick={onNewChat}
          disabled={disabled}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-md text-[13px] font-medium text-neutral-700 bg-white border border-neutral-200 hover:border-neutral-400 hover:text-neutral-900 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12h14" />
          </svg>
          New chat
        </button>
      </div>

      {/* stack info */}
      <div className="flex-1 overflow-y-auto px-3">
        <div className="mt-2 px-1">
          <div className="text-[10.5px] uppercase tracking-wider text-neutral-400 font-medium mb-2">
            Pipeline
          </div>
          <ul className="space-y-1.5">
            {STACK.map((row) => (
              <li
                key={row.label}
                className="flex items-baseline justify-between gap-3 text-[11.5px] leading-tight"
              >
                <span className="text-neutral-500 shrink-0">{row.label}</span>
                <span className="text-neutral-800 font-medium text-right truncate">
                  {row.value}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* footer */}
      <div className="border-t border-neutral-200 p-3 space-y-2">
        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 px-1 text-[11.5px] text-neutral-500 hover:text-neutral-900 transition-colors"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 .5C5.73.5.5 5.73.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.34-1.3-1.7-1.3-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.39.97.1-.75.4-1.27.73-1.56-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.46.11-3.05 0 0 .96-.31 3.15 1.18a10.9 10.9 0 0 1 5.74 0c2.18-1.49 3.14-1.18 3.14-1.18.63 1.59.24 2.76.12 3.05.74.81 1.18 1.84 1.18 3.1 0 4.42-2.69 5.39-5.25 5.67.41.36.78 1.05.78 2.13v3.16c0 .31.2.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z" />
          </svg>
          Source on GitHub
        </a>
        <div className="px-1 text-[10px] text-neutral-400">
          Portfolio demo · not affiliated with LangChain
        </div>
      </div>
    </aside>
  );
}