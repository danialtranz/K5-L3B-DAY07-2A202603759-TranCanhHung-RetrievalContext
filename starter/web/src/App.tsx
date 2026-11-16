import { useEffect, useMemo, useRef, useState } from "react";
import { getHealth, getMeta, postChat } from "./api";
import { ChatPanel } from "./components/ChatPanel";
import { HitList } from "./components/HitList";
import { PdfViewer } from "./components/PdfViewer";
import type {
  ChatMessage,
  HealthResponse,
  HitRecord,
  MetaResponse,
  StrategyId,
} from "./types";

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<StrategyId>("hybrid_wide");
  const [section, setSection] = useState("");
  const [topK, setTopK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeHits, setActiveHits] = useState<HitRecord[]>([]);
  const [selectedHit, setSelectedHit] = useState<HitRecord | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState(true);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    Promise.all([getHealth(), getMeta()])
      .then(([h, m]) => {
        setHealth(h);
        setMeta(m);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const countsLabel = useMemo(() => {
    if (!health) return "…";
    return Object.entries(health.collection_counts)
      .map(([k, v]) => `${k}:${v}`)
      .join(" · ");
  }, [health]);

  function selectHit(hit: HitRecord) {
    setSelectedHit(hit);
    setActiveHits((prev) => (prev.some((h) => h.chunk_id === hit.chunk_id) ? prev : [hit, ...prev]));
  }

  async function handleSend(question: string) {
    setError(null);
    setBusy(true);
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    const userId = uid();
    const assistantId = uid();
    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: question },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        strategy,
        hits: [],
        provider: "running…",
      },
    ]);

    try {
      const res = await postChat(
        {
          question,
          strategy,
          top_k: topK,
          section:
            strategy === "dense" || strategy === "hybrid_filter"
              ? section || undefined
              : undefined,
          generate: true,
        },
        ac.signal,
      );
      setActiveHits(res.hits || []);
      setSelectedHit(res.hits?.[0] ?? null);
      if (res.hits?.length) setSourcesOpen(true);
      const fromEvents = (res.agent_events || [])
        .filter((e) => e?.event === "search")
        .map((e) => (e.query || "").trim())
        .filter(Boolean);
      const queries =
        res.trace?.queries?.filter((q) => q.trim()).length
          ? res.trace.queries.map((q) => q.trim()).filter(Boolean)
          : fromEvents.length
            ? fromEvents
            : [res.question || question];
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId
            ? {
                ...msg,
                content: res.answer,
                strategy: res.strategy,
                hits: res.hits,
                references: res.references,
                provider: res.provider,
                trace: {
                  ...res.trace,
                  search_calls: Math.max(
                    res.trace?.search_calls ?? 0,
                    queries.length,
                    1,
                  ),
                  queries,
                },
              }
            : msg,
        ),
      );
    } catch (err) {
      if ((err as Error)?.name === "AbortError") return;
      setError(err instanceof Error ? err.message : String(err));
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId && !msg.content
            ? { ...msg, content: "Request failed.", provider: "error" }
            : msg,
        ),
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <div className="brand-mark">
            <span className="brand-dot" aria-hidden />
            <p className="eyebrow">Corpus · Day 07</p>
          </div>
          <h1>Agentic RAG</h1>
        </div>
        <div className="status">
          <span className={health?.llm_configured ? "ok" : undefined}>
            {health?.llm_configured ? health.chat_model || "LLM" : "retrieval"}
          </span>
          <span>{meta?.citation_style || "IEEE"}</span>
          <span>{meta?.pdf_available ? "PDF" : "no PDF"}</span>
          <span>{countsLabel}</span>
        </div>
      </header>

      {error ? <div className="banner error">{error}</div> : null}

      <main className={`layout ${sourcesOpen ? "sources-open" : "sources-closed"}`}>
        <div className="shell">
          <ChatPanel
            messages={messages}
            strategy={strategy}
            section={section}
            topK={topK}
            busy={busy}
            onStrategy={setStrategy}
            onSection={setSection}
            onTopK={setTopK}
            onSend={handleSend}
            onSelectMessageHits={(msg) => {
              setActiveHits(msg.hits || []);
              setSelectedHit(msg.hits?.[0] ?? null);
              if (msg.hits?.length) setSourcesOpen(true);
            }}
            onCiteClick={(hit) => {
              selectHit(hit);
              setSourcesOpen(true);
            }}
          />
        </div>

        <aside className="side pdf-side">
          <div className="shell side-block pdf-block">
            <div className="panel" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
              <PdfViewer hit={selectedHit} pdfUrl={meta?.pdf_url || "/api/corpus.pdf"} />
            </div>
          </div>
        </aside>

        {sourcesOpen ? (
          <aside className="shell sources-panel" aria-label="Sources">
            <div className="panel sources-panel-inner">
              <div className="sources-header">
                <h2>
                  Sources
                  {activeHits.length ? (
                    <span className="sources-count">{activeHits.length}</span>
                  ) : null}
                </h2>
                <button
                  type="button"
                  className="sources-toggle"
                  onClick={() => setSourcesOpen(false)}
                  aria-label="Hide sources"
                >
                  Hide
                </button>
              </div>
              <div className="sources-scroll">
                <HitList
                  hits={activeHits}
                  selectedId={selectedHit?.chunk_id ?? null}
                  onSelect={selectHit}
                />
              </div>
            </div>
          </aside>
        ) : (
          <button
            type="button"
            className="sources-rail"
            onClick={() => setSourcesOpen(true)}
            aria-label="Show sources"
            title="Show sources"
          >
            <span>Sources</span>
            {activeHits.length ? <em>{activeHits.length}</em> : null}
          </button>
        )}
      </main>
    </div>
  );
}
