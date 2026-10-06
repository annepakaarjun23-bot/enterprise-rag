import type { ChatStatus } from "../api/types";

const LABELS: Record<ChatStatus, string> = {
  idle: "",
  retrieving: "Retrieving context…",
  generating: "Generating answer…",
  error: "Error",
};

export function StatusBadge({ status }: { status: ChatStatus }) {
  if (status === "idle") return null;
  const busy = status === "retrieving" || status === "generating";
  return (
    <div className="flex items-center gap-2 text-xs text-neutral-500">
      {busy && (
        <span className="inline-block size-2 rounded-full bg-blue-500 animate-pulse" />
      )}
      <span>{LABELS[status]}</span>
    </div>
  );
}