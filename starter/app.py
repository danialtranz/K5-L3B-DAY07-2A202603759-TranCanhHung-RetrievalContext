"""Day 7 FastAPI: agentic RAG chat via run_agent + IEEE citations for the FE."""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic import run_agent  # noqa: E402
from llm_controller import OpenAICompatController  # noqa: E402
from retrieval.chroma_store import DenseChromaRetriever  # noqa: E402
from retrieval.embeddings import DEFAULT_MODEL, MiniLMEmbedder  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402
from retrieval.tfidf import TfidfRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger('day7.app')

load_dotenv(ROOT / '.env')

SQLITE_PATH = Path(os.getenv('SQLITE_PATH', str(ROOT / 'data' / 'lab.db')))
CHROMA_PATH = Path(os.getenv('CHROMA_PATH', str(ROOT / 'data' / 'chroma')))
FIGURES_DIR = Path(os.getenv('FIGURES_DIR', str(ROOT / 'data' / 'figures')))
CHUNKS_PATH = Path(os.getenv('CHUNKS_PATH', str(ROOT / 'data' / 'converted' / 'chunks.json')))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', DEFAULT_MODEL)
CHROMA_STRATEGY = os.getenv('CHROMA_STRATEGY', 'C').upper()
WEB_DIST = ROOT / 'web' / 'dist'
DOC_IEEE_SHORT = os.getenv(
    'DOC_IEEE_SHORT',
    'S. Rahimi and F. Haug, Distributed Database Management Systems: A Practical Approach, Wiley-IEEE, 2010',
)

STATE: dict[str, Any] = {}


def _resolve_pdf_path() -> Path | None:
    raw = os.getenv('PDF_PATH', '').strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.is_file():
            return p
    for base in (REPO_ROOT / 'assets', ROOT.parent.parent / 'K5-L3B-Day06-TranCanhHung-2A202603759-DataIngestionEngineering' / 'assets'):
        if base.is_dir():
            candidates = sorted(base.glob('*.pdf'))
            if candidates:
                return candidates[0]
    fallback = ROOT / 'data' / 'corpus.pdf'
    return fallback if fallback.is_file() else None


def _sse(event: str, data: dict[str, Any] | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f'event: {event}\ndata: {payload}\n\n'


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: str = Field(default='hybrid_wide')
    top_k: int = Field(default=5, ge=1, le=10)
    section: str | None = None
    generate: bool = True
    max_searches: int = Field(default=4, ge=1, le=8)
    budget: int = Field(default=5000, ge=200, le=8000)


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    strategy: str = Field(default='hybrid_wide')
    top_k: int = Field(default=5, ge=1, le=10)
    section: str | None = None


class MethodRetriever:
    """Retriever adapter for agentic.run_agent over Day 7 retrieval methods."""

    def __init__(self, method: str) -> None:
        self.method = _normalize_method(method)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        dense: DenseChromaRetriever = STATE['dense']
        lexical: TfidfRetriever = STATE['lexical']
        hybrid: HybridRetriever = STATE['hybrid']
        hybrid_wide: HybridRetriever = STATE['hybrid_wide']
        if self.method == 'tfidf':
            return lexical.search(query, filters=None, top_k=top_k)
        if self.method == 'dense':
            return dense.search(query, filters=filters, top_k=top_k)
        if self.method == 'hybrid':
            return hybrid.search(query, filters=None, top_k=top_k)
        if self.method == 'hybrid_filter':
            return hybrid.search(query, filters=filters, top_k=top_k)
        return hybrid_wide.search(query, filters=None, top_k=top_k)


def _normalize_method(strategy: str) -> str:
    s = (strategy or 'hybrid_wide').strip().lower()
    # Back-compat with Day 6 FE labels A/B/C → dense on collection C (or A/B)
    if s in {'a', 'b', 'c'}:
        return 'dense'
    if s in {'dense', 'tfidf', 'hybrid', 'hybrid_wide', 'hybrid_filter'}:
        return s
    return 'hybrid_wide'


def _filters(section: str | None, method: str) -> dict[str, Any] | None:
    if method in {'dense', 'hybrid_filter'} and section and section.strip():
        return {'section': section.strip()}
    return None


def _figure_image_url(image_path: str | None, figure_id: str | None) -> str | None:
    candidates: list[Path] = []
    if image_path:
        p = Path(image_path)
        candidates.append(p)
        candidates.append(FIGURES_DIR / p.name)
    if figure_id:
        candidates.append(FIGURES_DIR / f'{figure_id}.png')
    for cand in candidates:
        try:
            if cand.is_file():
                return f'/api/figures/{cand.name}'
        except OSError:
            continue
    if figure_id and FIGURES_DIR.exists():
        for f in FIGURES_DIR.glob(f'{figure_id}*.png'):
            return f'/api/figures/{f.name}'
    return None


def _ieee_citation(n: int, meta: dict[str, Any], text: str) -> str:
    page = meta.get('page')
    section = (meta.get('section') or '').strip()
    chunk_type = (meta.get('chunk_type') or 'text').strip()
    locator_parts: list[str] = []
    if chunk_type == 'figure':
        head = (text or '').strip().split('\n', 1)[0]
        label = head.split('.', 1)[0].strip() if head.lower().startswith('figure') else 'Fig.'
        locator_parts.append(label)
    elif chunk_type == 'table':
        head = (text or '').strip().split('\n', 1)[0]
        locator_parts.append(head.split(',', 1)[0].strip()[:40] if head.upper().startswith('TABLE') else 'Table')
    elif section:
        locator_parts.append(f'sec. {section}')
    if page not in (None, ''):
        locator_parts.append(f'p. {page}')
    locator = ', '.join(locator_parts) if locator_parts else 'retrieved chunk'
    return f'[{n}] {DOC_IEEE_SHORT}, {locator}.'


def _to_api_hit(rank: int, hit: dict[str, Any]) -> dict[str, Any]:
    meta = dict(hit.get('metadata') or {})
    if hit.get('source_id') and not meta.get('document_id'):
        meta['document_id'] = hit['source_id']
    if hit.get('source') and not meta.get('source'):
        meta['source'] = hit['source']
    if hit.get('page') is not None and meta.get('page') is None:
        meta['page'] = hit['page']
    return {
        'rank': rank,
        'chunk_id': hit.get('id') or hit.get('chunk_id'),
        'text': hit.get('text') or '',
        'score': hit.get('score'),
        'metadata': meta,
    }


def _enrich_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conn = None
    if SQLITE_PATH.exists():
        conn = sqlite3.connect(str(SQLITE_PATH))
        conn.row_factory = sqlite3.Row
    out: list[dict[str, Any]] = []
    try:
        for h in hits:
            row = dict(h)
            meta = dict(row.get('metadata') or {})
            cid = h.get('chunk_id')
            image_path = meta.get('image_path') or ''
            figure_id = meta.get('figure_id') or ''
            if conn and cid:
                cur = conn.execute(
                    """
                    SELECT c.chunk_id, c.strategy_id, c.document_id, c.source, c.page, c.text,
                           c.chunk_type, c.section, c.figure_id, c.table_id, c.char_count,
                           f.image_path AS figure_image_path, f.figure_label, f.caption
                    FROM chunks c
                    LEFT JOIN figures f ON f.figure_id = c.figure_id
                    WHERE c.chunk_id = ?
                    """,
                    (cid,),
                )
                db_row = cur.fetchone()
                if db_row:
                    payload = dict(db_row)
                    row['sqlite'] = payload
                    if payload.get('figure_image_path'):
                        image_path = str(payload['figure_image_path'])
                    if payload.get('figure_id'):
                        figure_id = str(payload['figure_id'])
                    for key in ('chunk_type', 'section', 'page', 'figure_label'):
                        if payload.get(key) is not None:
                            meta[key] = payload[key]
                    if not row.get('text') and payload.get('text'):
                        row['text'] = payload['text']
            n = int(row.get('rank') or len(out) + 1)
            row['metadata'] = meta
            row['cite_num'] = n
            row['citation_ieee'] = _ieee_citation(n, meta, row.get('text') or '')
            image_url = _figure_image_url(
                str(image_path) if image_path else None,
                str(figure_id) if figure_id else None,
            )
            row['image_url'] = image_url
            row['image_ref'] = image_url
            out.append(row)
    finally:
        if conn is not None:
            conn.close()
    return out


def _references_block(hits: list[dict[str, Any]]) -> str:
    lines = [str(h.get('citation_ieee') or '') for h in hits if h.get('citation_ieee')]
    if not lines:
        return ''
    return 'References\n' + '\n'.join(lines)


def _context_answer(question: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return 'No relevant passages were retrieved for this question.'
    parts = [f'Based on the retrieved evidence for: {question}', '']
    for h in hits:
        n = h.get('cite_num') or h.get('rank')
        excerpt = (h.get('text') or '').strip().replace('\n', ' ')
        parts.append(f'[{n}] {excerpt}')
        if h.get('image_url'):
            parts.append(f'    Image ref: {h["image_url"]}')
        parts.append('')
    refs = _references_block(hits)
    if refs:
        parts.append(refs)
    return '\n'.join(parts).strip()


def _chunks_from_agent_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get('event') != 'search':
            continue
        for row in event.get('results') or []:
            cid = row.get('id')
            if isinstance(cid, str) and cid:
                by_id[cid] = row
    return by_id


def _ordered_chunk_ids(result: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for cid in result.get('citations') or []:
        if isinstance(cid, str) and cid in by_id and cid not in ordered:
            ordered.append(cid)
    for event in reversed(result.get('events') or []):
        if event.get('event') != 'search':
            continue
        for cid in event.get('context_ids') or []:
            if isinstance(cid, str) and cid in by_id and cid not in ordered:
                ordered.append(cid)
        for row in event.get('results') or []:
            cid = row.get('id')
            if isinstance(cid, str) and cid in by_id and cid not in ordered:
                ordered.append(cid)
        break
    if not ordered:
        ordered = list(by_id.keys())
    return ordered


def _rewrite_answer_ieee(answer: str, hits: list[dict[str, Any]], *, attach_refs: bool) -> str:
    text = answer or ''
    for hit in sorted(hits, key=lambda h: len(str(h.get('chunk_id') or '')), reverse=True):
        cid = hit.get('chunk_id')
        num = hit.get('cite_num')
        if not cid or num is None:
            continue
        text = text.replace(f'[{cid}]', f'[{num}]')
    # Drop leftover long chunk-id markers if any
    text = re.sub(r'\[C_[^\]]+\]', '', text)
    if attach_refs:
        refs = _references_block(hits)
        if refs and not re.search(r'\nReferences\s*\n', text, flags=re.I):
            text = text.rstrip() + '\n\n' + refs
    return text.strip()


def _agent_to_fe_payload(
    *,
    question: str,
    method: str,
    section: str | None,
    top_k: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    events = list(result.get('events') or [])
    by_id = _chunks_from_agent_events(events)
    ordered_ids = _ordered_chunk_ids(result, by_id)
    api_hits = []
    for i, cid in enumerate(ordered_ids, start=1):
        api_hits.append(_to_api_hit(i, by_id[cid]))
    hits = _enrich_hits(api_hits)
    status = result.get('status')
    answered = status == 'answered'
    answer = _rewrite_answer_ieee(
        str(result.get('answer') or ''),
        hits,
        attach_refs=answered and bool(result.get('citations')),
    )
    search_events = [e for e in events if e.get('event') == 'search']
    search_n = len(search_events)
    queries = [str(e.get('query') or '').strip() for e in search_events if str(e.get('query') or '').strip()]
    # Slim event list for FE (full context stays out of the wire payload).
    agent_events_lite = [
        {
            'event': e.get('event'),
            'query': e.get('query'),
            'result_count': len(e.get('results') or []) if e.get('event') == 'search' else None,
            'context_ids': e.get('context_ids') if e.get('event') == 'search' else None,
        }
        for e in events
        if e.get('event') in {'search', 'answer', 'search_limit', 'invalid_or_missing_citations'}
    ]
    provider = f"agent:{os.getenv('CHAT_MODEL', '').strip() or 'llm'}"
    return {
        'question': question,
        'strategy': method,
        'section_filter': section,
        'top_k': top_k,
        'answer': answer,
        'provider': provider,
        'status': status,
        'citations': result.get('citations') or [],
        'hits': hits,
        'references': (
            [h.get('citation_ieee') for h in hits if h.get('citation_ieee')] if answered else []
        ),
        'agent_events': agent_events_lite,
        'trace': {
            'stages': ['run_agent', 'tool_search_x' + str(search_n), 'final_answer', 'ieee_map'],
            'hit_count': len(hits),
            'search_calls': search_n,
            'queries': queries,
            'status': status,
        },
    }


def _run_agent_chat(req: ChatRequest) -> dict[str, Any]:
    method = _normalize_method(req.strategy)
    filters = _filters(req.section, method)
    controller = OpenAICompatController()
    retriever = MethodRetriever(method)
    result = run_agent(
        req.question,
        controller,
        retriever,
        filters=filters,
        max_searches=req.max_searches,
        top_k=req.top_k,
        budget=req.budget,
    )
    return _agent_to_fe_payload(
        question=req.question,
        method=method,
        section=req.section,
        top_k=req.top_k,
        result=result,
    )


def _retrieve(question: str, strategy: str, top_k: int, section: str | None) -> list[dict[str, Any]]:
    method = _normalize_method(strategy)
    filters = _filters(section, method)
    if method == 'hybrid_filter':
        filters = {'section': section.strip()} if section and section.strip() else None

    dense: DenseChromaRetriever = STATE['dense']
    lexical: TfidfRetriever = STATE['lexical']
    hybrid: HybridRetriever = STATE['hybrid']
    hybrid_wide: HybridRetriever = STATE['hybrid_wide']

    if method == 'tfidf':
        raw = lexical.search(question, filters=None, top_k=top_k)
    elif method == 'dense':
        raw = dense.search(question, filters=filters, top_k=top_k)
    elif method == 'hybrid':
        raw = hybrid.search(question, filters=None, top_k=top_k)
    elif method == 'hybrid_filter':
        raw = hybrid.search(question, filters=filters, top_k=top_k)
    else:
        raw = hybrid_wide.search(question, filters=None, top_k=top_k)

    api_hits = [_to_api_hit(i, h) for i, h in enumerate(raw, start=1)]
    return _enrich_hits(api_hits)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Loading embedder: %s', EMBEDDING_MODEL)
    embedder = MiniLMEmbedder(EMBEDDING_MODEL)
    if not CHUNKS_PATH.exists():
        raise RuntimeError(f'Missing chunks JSON: {CHUNKS_PATH}')
    chunks = json.loads(CHUNKS_PATH.read_text(encoding='utf-8'))
    dense = DenseChromaRetriever(CHROMA_PATH, embedder, strategy=CHROMA_STRATEGY)
    lexical = TfidfRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical, candidate_k=8)
    hybrid_wide = HybridRetriever(dense, lexical, candidate_k=12)
    STATE.update(
        {
            'embedder': embedder,
            'dense': dense,
            'lexical': lexical,
            'hybrid': hybrid,
            'hybrid_wide': hybrid_wide,
            'chunks': chunks,
        }
    )
    logger.info('Ready. chroma=%s sqlite=%s method_default=hybrid_wide', CHROMA_PATH, SQLITE_PATH)
    yield
    STATE.clear()


app = FastAPI(title='Day07 Agentic RAG Chat', version='0.1.0', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5173',
        'http://127.0.0.1:5173',
        'http://localhost:4173',
        'http://127.0.0.1:4173',
        'http://localhost:8000',
        'http://127.0.0.1:8000',
    ],
    allow_origin_regex=r'http://(localhost|127\.0\.0\.1):\d+',
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/api/health')
def health() -> dict[str, Any]:
    load_dotenv(ROOT / '.env', override=True)
    dense: DenseChromaRetriever = STATE['dense']
    try:
        count = dense.collection.count()
    except Exception as exc:  # noqa: BLE001
        count = f'error: {exc}'
    return {
        'ok': True,
        'embedding_model': EMBEDDING_MODEL,
        'dimensions': STATE['embedder'].dimensions,
        'chroma_path': str(CHROMA_PATH),
        'sqlite_path': str(SQLITE_PATH),
        'collection_counts': {CHROMA_STRATEGY: count},
        'llm_configured': bool(
            os.getenv('OPENROUTER_API_KEY', '').strip() or os.getenv('OPENAI_API_KEY', '').strip()
        ),
        'chat_model': os.getenv('CHAT_MODEL', '').strip() or None,
    }


@app.get('/api/meta')
def meta() -> dict[str, Any]:
    return {
        'strategies': [
            {'id': 'dense', 'label': 'Dense (Chroma C)', 'filter': True},
            {'id': 'tfidf', 'label': 'TF-IDF lexical', 'filter': False},
            {'id': 'hybrid', 'label': 'Hybrid RRF', 'filter': False},
            {'id': 'hybrid_wide', 'label': 'Hybrid wide (selected)', 'filter': False},
            {'id': 'hybrid_filter', 'label': 'Hybrid + section filter', 'filter': True},
        ],
        'hit_fields': [
            'rank',
            'cite_num',
            'citation_ieee',
            'chunk_id',
            'score',
            'text',
            'image_url',
            'metadata.page',
            'metadata.section',
            'metadata.chunk_type',
        ],
        'citation_style': 'IEEE',
        'viz_ready': True,
        'pdf_url': '/api/corpus.pdf',
        'stream_url': '/api/chat/stream',
        'pdf_available': _resolve_pdf_path() is not None,
    }


@app.get('/api/figures/{filename}')
def get_figure(filename: str) -> FileResponse:
    safe = Path(filename).name
    path = FIGURES_DIR / safe
    if not path.is_file():
        raise HTTPException(404, f'figure not found: {safe}')
    return FileResponse(path, media_type='image/png')


@app.get('/api/corpus.pdf')
def get_corpus_pdf() -> FileResponse:
    path = _resolve_pdf_path()
    if path is None:
        raise HTTPException(404, 'corpus PDF not found (set PDF_PATH or place PDF under assets/)')
    return FileResponse(
        path,
        media_type='application/pdf',
        filename=path.name,
        headers={'Content-Disposition': f'inline; filename="{path.name}"'},
    )


@app.post('/api/search')
def search(req: SearchRequest) -> dict[str, Any]:
    method = _normalize_method(req.strategy)
    hits = _retrieve(req.question, req.strategy, req.top_k, req.section)
    return {
        'question': req.question,
        'strategy': method,
        'section_filter': req.section,
        'top_k': req.top_k,
        'hits': hits,
        'references': [h.get('citation_ieee') for h in hits if h.get('citation_ieee')],
    }


@app.post('/api/chat')
def chat(req: ChatRequest) -> dict[str, Any]:
    """Final agent response only (no token streaming). Uses starter/agentic.run_agent."""
    if not req.question.strip():
        raise HTTPException(400, 'question is empty')
    if not req.generate:
        hits = _retrieve(req.question, req.strategy, req.top_k, req.section)
        return {
            'question': req.question,
            'strategy': _normalize_method(req.strategy),
            'section_filter': req.section,
            'top_k': req.top_k,
            'answer': _context_answer(req.question, hits),
            'provider': 'retrieval_only',
            'status': 'answered' if hits else 'abstain',
            'citations': [],
            'hits': hits,
            'references': [h.get('citation_ieee') for h in hits if h.get('citation_ieee')],
            'trace': {
                'stages': ['retrieve_only'],
                'hit_count': len(hits),
                'search_calls': 1,
                'queries': [req.question.strip()],
            },
        }
    try:
        return _run_agent_chat(req)
    except Exception as exc:  # noqa: BLE001
        logger.exception('run_agent failed')
        raise HTTPException(502, f'agent failed: {exc}') from exc


@app.post('/api/chat/stream')
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE compatibility: only emits the final `done` payload (no token deltas)."""
    if not req.question.strip():
        raise HTTPException(400, 'question is empty')

    def event_gen() -> Iterator[str]:
        try:
            payload = chat(req)
            # Single final event for the FE — no per-token streaming.
            yield _sse('done', payload)
        except HTTPException as exc:
            yield _sse('error', {'message': exc.detail})
        except Exception as exc:  # noqa: BLE001
            logger.exception('chat_stream failed')
            yield _sse('error', {'message': str(exc)})

    return StreamingResponse(
        event_gen(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )


if WEB_DIST.exists():
    assets = WEB_DIST / 'assets'
    if assets.exists():
        app.mount('/assets', StaticFiles(directory=assets), name='assets')

    @app.get('/')
    def index() -> FileResponse:
        return FileResponse(WEB_DIST / 'index.html')


def main() -> None:
    import uvicorn

    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True, app_dir=str(ROOT))


if __name__ == '__main__':
    main()
