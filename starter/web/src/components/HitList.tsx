import { useState, type MouseEvent } from "react";
import type { HitRecord } from "../types";

type Props = {
  hits: HitRecord[];
  selectedId: string | null;
  onSelect: (hit: HitRecord) => void;
};

async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  document.body.removeChild(ta);
}

export function HitList({ hits, selectedId, onSelect }: Props) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  if (!hits.length) {
    return <p className="muted">No sources yet.</p>;
  }

  async function handleCopy(hit: HitRecord, e: MouseEvent) {
    e.stopPropagation();
    const text = hit.text?.trim();
    if (!text) return;
    try {
      await copyText(text);
      setCopiedId(hit.chunk_id);
      window.setTimeout(() => {
        setCopiedId((cur) => (cur === hit.chunk_id ? null : cur));
      }, 1400);
    } catch {
      setCopiedId(null);
    }
  }

  return (
    <ul className="hit-list">
      {hits.map((hit) => {
        const meta = hit.metadata || {};
        const active = hit.chunk_id === selectedId;
        const n = hit.cite_num ?? hit.rank;
        const score = typeof hit.score === "number" ? hit.score.toFixed(3) : "—";
        const img = hit.image_url || hit.image_ref;
        const copied = copiedId === hit.chunk_id;
        return (
          <li key={hit.chunk_id}>
            <div
              role="button"
              tabIndex={0}
              className={`hit-card ${active ? "active" : ""}`}
              onClick={() => onSelect(hit)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelect(hit);
                }
              }}
            >
              <div className="hit-card-top">
                <span className="badge cite">[{n}]</span>
                <span className="badge soft">{meta.chunk_type || "text"}</span>
                <span className="score">{score}</span>
                <button
                  type="button"
                  className={`hit-copy ${copied ? "copied" : ""}`}
                  onClick={(e) => void handleCopy(hit, e)}
                  aria-label={copied ? "Copied" : "Copy chunk text"}
                  title={copied ? "Copied" : "Copy chunk"}
                >
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
              <div className="hit-meta">
                p.{meta.page ?? "—"} · §{meta.section || "—"}
              </div>
              <p className="hit-cite">{hit.citation_ieee}</p>
              <p className="hit-preview">{hit.text || ""}</p>
              {img ? (
                <img className="hit-thumb" src={img} alt={meta.figure_label || `figure ${n}`} />
              ) : null}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
