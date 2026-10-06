import type { Message as MessageType } from "../api/types";
import { Markdown } from "./Markdown";
import { SourceList } from "./SourceList";

export function Message({ message }: { message: MessageType }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-md bg-neutral-100 text-neutral-900 px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap break-words">
          {message.content}
        </div>
      </div>
    );
  }

  const isStreaming = message.streaming === true;
  const hasContent = message.content.length > 0;

  return (
    <div className="flex justify-start">
      <div className="w-full">
        {hasContent ? (
          <div className={isStreaming ? "streaming-cursor" : ""}>
            <Markdown>{message.content}</Markdown>
          </div>
        ) : isStreaming ? (
          <div className="flex items-center gap-1 py-1 text-neutral-400">
            <span className="thinking-dot inline-block size-1.5 rounded-full bg-neutral-400" />
            <span className="thinking-dot inline-block size-1.5 rounded-full bg-neutral-400" />
            <span className="thinking-dot inline-block size-1.5 rounded-full bg-neutral-400" />
          </div>
        ) : (
          <div className="text-[15px] text-neutral-400 italic">
            No answer produced.
          </div>
        )}

        {message.error && (
          <div className="mt-3 text-[13px] text-red-700 bg-red-50 border border-red-100 rounded-md px-3 py-2">
            {message.error}
          </div>
        )}

        {!isStreaming && message.chunks && message.chunks.length > 0 && (
          <SourceList chunks={message.chunks} />
        )}

        {!isStreaming && message.durationMs !== undefined && (
          <div className="mt-2 text-[11px] text-neutral-400 tabular-nums">
            {(message.durationMs / 1000).toFixed(1)}s
          </div>
        )}
      </div>
    </div>
  );
}