/** Shared API types — keep hit shape stable for visualization + IEEE cites. */

export type StrategyId = "dense" | "tfidf" | "hybrid" | "hybrid_wide" | "hybrid_filter";

export interface HitMetadata {
  document_id?: string;
  source?: string;
  page?: number;
  section?: string;
  chunk_type?: string;
  strategy_id?: string;
  figure_id?: string;
  table_id?: string;
  figure_label?: string;
  image_path?: string;
  [key: string]: string | number | boolean | undefined;
}

export interface HitRecord {
  rank: number;
  cite_num?: number;
  citation_ieee?: string;
  chunk_id: string;
  text: string;
  score: number | null;
  metadata: HitMetadata;
  image_url?: string | null;
  image_ref?: string | null;
  sqlite?: Record<string, string | number | null>;
}

export interface ChatResponse {
  question: string;
  strategy: StrategyId;
  section_filter: string | null;
  top_k: number;
  answer: string;
  provider: string;
  status?: string;
  citations?: string[];
  hits: HitRecord[];
  references?: string[];
  agent_events?: { event?: string; query?: string }[];
  trace: {
    stages: string[];
    hit_count: number;
    search_calls?: number;
    queries?: string[];
    status?: string;
  };
}

export interface HealthResponse {
  ok: boolean;
  embedding_model: string;
  dimensions: number;
  collection_counts: Record<string, number | string>;
  llm_configured: boolean;
  chat_model?: string | null;
  chroma_path: string;
  sqlite_path: string;
}

export interface MetaResponse {
  strategies: { id: StrategyId; label: string; filter: boolean }[];
  hit_fields: string[];
  citation_style?: string;
  viz_ready: boolean;
  pdf_url?: string;
  stream_url?: string;
  pdf_available?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  strategy?: StrategyId;
  hits?: HitRecord[];
  references?: string[];
  provider?: string;
  trace?: ChatResponse["trace"];
}
