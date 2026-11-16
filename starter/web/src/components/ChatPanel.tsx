import { useEffect, useRef, type FormEvent } from "react";
import type { ChatMessage, HitRecord, StrategyId } from "../types";
import { CitedAnswer } from "./CitedAnswer";

type Props = {
  messages: ChatMessage[];
  strategy: StrategyId;
  section: string;
  topK: number;
  busy: boolean;
  onStrategy: (s: StrategyId) => void;
  onSection: (s: string) => void;
  onTopK: (n: number) => void;
  onSend: (question: string) => void;
  onSelectMessageHits: (msg: ChatMessage) => void;
  onCiteClick: (hit: HitRecord) => void;
};

const NEEDS_SECTION: StrategyId[] = ["dense", "hybrid_filter"];

function searchQueriesFor(messages: ChatMessage[], msg: ChatMessage): string[] {
  const fromTrace = msg.trace?.queries?.map((q) => q.trim()).filter(Boolean) ?? [];
  if (fromTrace.length) return fromTrace;
  if (!msg.trace?.search_calls) return [];
  const idx = messages.findIndex((x) => x.id === msg.id);
  const prevUser = [...messages.slice(0, idx)].reverse().find((x) => x.role === "user");
  return [prevUser?.content?.trim() || "—"];
}

export function ChatPanel({
  messages,
  strategy,
  section,
  topK,
  busy,
  onStrategy,
  onSection,
  onTopK,
  onSend,
  onSelectMessageHits,
  onCiteClick,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const sectionEnabled = NEEDS_SECTION.includes(strategy);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const q = inputRef.current?.value.trim() || "";
    if (!q || busy) return;
    onSend(q);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <section className="panel chat-panel">
      <header className="toolbar">
        <label>
          Retrieval
          <select
            value={strategy}
            onChange={(e) => onStrategy(e.target.value as StrategyId)}
            disabled={busy}
          >
            <option value="hybrid_wide">Hybrid wide</option>
            <option value="hybrid">Hybrid RRF</option>
            <option value="hybrid_filter">Hybrid + section</option>
            <option value="dense">Dense</option>
            <option value="tfidf">TF-IDF</option>
          </select>
        </label>
        <label>
          Section
          <input
            value={section}
            onChange={(e) => onSection(e.target.value)}
            placeholder={sectionEnabled ? "5.3.1" : "—"}
            disabled={busy || !sectionEnabled}
          />
        </label>
        <label>
          Top-k
          <input
            type="number"
            min={1}
            max={10}
            value={topK}
            onChange={(e) => onTopK(Number(e.target.value) || 5)}
            disabled={busy}
          />
        </label>
      </header>

      <div className="messages">
        {messages.length === 0 ? (
          <div className="empty-state">Ask a question about the textbook corpus.</div>
        ) : null}
        {messages.map((m) => (
          <article
            key={m.id}
            className={`bubble ${m.role}`}
            onClick={() => m.role === "assistant" && m.hits && onSelectMessageHits(m)}
          >
            <div className="bubble-meta">
              {m.role === "assistant" ? (
                <>
                  <span>{m.provider || "assistant"}</span>
                  {m.strategy ? <span>{m.strategy}</span> : null}
                  {m.hits ? <span>{m.hits.length} sources</span> : null}
                  {m.trace?.search_calls != null ? (
                    <span>{m.trace.search_calls} search{m.trace.search_calls === 1 ? "" : "es"}</span>
                  ) : null}
                </>
              ) : (
                <span>you</span>
              )}
            </div>
            {m.role === "assistant" ? (
              m.content ? (
                <>
                  <CitedAnswer
                    content={m.content}
                    hits={m.hits}
                    references={m.references}
                    onCiteClick={onCiteClick}
                  />
                  {(() => {
                    const shown = searchQueriesFor(messages, m);
                    if (!shown.length) return null;
                    const n = m.trace?.search_calls ?? shown.length;
                    return (
                      <div className="search-trace">
                        <div className="search-trace-title">
                          Searched {n} time{n === 1 ? "" : "s"}
                        </div>
                        <ol>
                          {shown.map((q, i) => (
                            <li key={`${m.id}-q-${i}`}>
                              <span className="search-trace-n">#{i + 1}</span>
                              <code>{q}</code>
                            </li>
                          ))}
                        </ol>
                      </div>
                    );
                  })()}
                </>
              ) : (
                <p className="busy-line">Searching…</p>
              )
            ) : (
              <pre className="bubble-body">{m.content}</pre>
            )}
          </article>
        ))}
        {busy && messages.every((m) => m.role !== "assistant" || m.content) ? (
          <p className="busy-line">Working…</p>
        ) : null}
        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          rows={2}
          placeholder="Ask the corpus…"
          disabled={busy}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <button type="submit" disabled={busy}>
          Send
        </button>
      </form>
    </section>
  );
}
