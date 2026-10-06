export type RetrievalMode = "dense" | "hybrid";

export interface ChatOptions {
  mode: RetrievalMode;
  topK: number;
  automerge: boolean;
}

export interface ChatChunk {
  chunk_id: string;
  parent_chunk_id: string;
  text: string;
  score: number;
  library: string;
  version: string;
  source_path: string;
  source_url: string;
  heading_path: string;
  has_code: boolean;
}

// The custom-stream events emitted by the FastAPI /chat/stream route
export type SSEEvent =
  | { type: "session"; session_id: string }
  | { type: "stage"; node: string; message: string }
  | { type: "stage_done"; node: string }
  | { type: "chunk"; chunk: ChatChunk }
  | { type: "token"; text: string }
  | { type: "error"; node?: string; message: string }
  | {
      type: "done";
      session_id: string;
      answer: string;
      chunks: ChatChunk[];
      duration_ms: number;
    };

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  chunks?: ChatChunk[];
  streaming?: boolean;
  durationMs?: number;
  error?: string;
}

export type ChatStatus = "idle" | "retrieving" | "generating" | "error";