import { useState } from "react";
import type { ChatChunk } from "../api/types";

function labelFor(chunk: ChatChunk): string {
  const lib = `${chunk.library}/${chunk.version}`;
  const path = chunk.heading_path || chunk.source_path || chunk.chunk_id.slice(0, 8);
  return `${lib} · ${path}`;
}

export function SourceList({ chunks }: { chunks: ChatChunk[] }) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<number | null>(null);

  if (!chunks.length) return null;

  return (
    <div className="mt-3 border border-neutral-200 rounded-lg overflow-hidden bg-white">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-neutral-600 hover:bg-neutral-50"
      >
        <span>
          {chunks.length} source{chunks.length === 1 ? "" : "s"}
        </span>
        <span className="text-neutral-400">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <ul className="divide-y divide-neutral-100">
          {chunks.map((c, i) => {
            const isOpen = expanded === i;
            return (
              <li key={c.chunk_id} className="text-xs">
                <button
                  type="button"
                  onClick={() => setExpanded(isOpen ? null : i)}
                  className="w-full text-left px-3 py-2 hover:bg-neutral-50"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-[11px] text-neutral-500">
                      [S{i + 1}]
                    </span>
                    <span className="flex-1 truncate text-neutral-700">
                      {labelFor(c)}
                    </span>
                    <span className="text-neutral-400 tabular-nums">
                      {c.score.toFixed(2)}
                    </span>
                  </div>
                </button>
                {isOpen && (
                  <div className="px-3 pb-3 text-neutral-600">
                    <pre className="whitespace-pre-wrap break-words font-sans text-[12.5px] leading-relaxed">
                      {c.text}
                    </pre>
                    {c.source_url && (
                      <a
                        href={c.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="mt-2 inline-block text-blue-600 hover:underline"
                      >
                        Open source ↗
                      </a>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}