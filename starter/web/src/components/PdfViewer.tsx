import { useMemo } from "react";
import type { HitRecord } from "../types";

type Props = {
  hit: HitRecord | null;
  pdfUrl?: string;
};

export function PdfViewer({ hit, pdfUrl = "/api/corpus.pdf" }: Props) {
  const page = hit?.metadata?.page != null ? Number(hit.metadata.page) : null;
  const section = hit?.metadata?.section || "—";
  const n = hit?.cite_num ?? hit?.rank;

  const src = useMemo(() => {
    if (page && Number.isFinite(page) && page > 0) {
      return `${pdfUrl}#page=${page}`;
    }
    return pdfUrl;
  }, [pdfUrl, page]);

  return (
    <div className="pdf-viewer">
      <header className="pdf-toolbar">
        <div>
          <strong>PDF</strong>
          {hit ? (
            <span className="muted">
              {" "}
              · [{n}] · p.{page ?? "—"} · §{section}
            </span>
          ) : null}
        </div>
        <a className="pdf-open" href={page ? src : pdfUrl} target="_blank" rel="noreferrer">
          {page ? `Open p.${page}` : "Open"}
        </a>
      </header>
      <iframe key={src} className="pdf-frame" title="Corpus PDF" src={src} />
      {hit?.text ? (
        <details className="pdf-passage">
          <summary>Passage</summary>
          <pre>{hit.text}</pre>
        </details>
      ) : null}
    </div>
  );
}
