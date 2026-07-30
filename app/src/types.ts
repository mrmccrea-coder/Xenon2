// Shared frontend types for Phase 3's chat UI.
//
// Conversations are held purely in-memory for this phase (Sidebar.vue's job per the Phase 3
// spec) -- real persistence to disk is Phase 5's job, not built here.

export type Role = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  /** True while an assistant reply is still streaming in token-by-token. */
  streaming?: boolean;
  /** True if generation ended in an error (message.content holds the error text). */
  errored?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number; // epoch ms, used for Today/Yesterday/Older grouping
  messages: ChatMessage[];
}

export function newId(): string {
  return crypto.randomUUID();
}
