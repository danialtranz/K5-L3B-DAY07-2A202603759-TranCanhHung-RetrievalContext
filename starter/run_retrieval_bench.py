"""Benchmark dense / TF-IDF / hybrid retrieval on Day 7 questions."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

_STARTER = Path(__file__).resolve().parent
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))

from baseline import hit, pack_context  # noqa: E402
from retrieval.chroma_store import DenseChromaRetriever  # noqa: E402
from retrieval.embeddings import MiniLMEmbedder  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402
from retrieval.tfidf import TfidfRetriever  # noqa: E402


Searcher = Callable[..., list[dict[str, Any]]]


def evaluate(
    name: str,
    search_fn: Searcher,
    questions: list[dict],
    *,
    top_k: int,
    budget: int,
    use_question_filters: bool = True,
) -> tuple[list[dict], dict]:
    traces = []
    for q in questions:
        filters = q.get('filter') if use_question_filters else None
        start = time.perf_counter()
        results = search_fn(q['question'], filters=filters, top_k=top_k)
        context, ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        traces.append(
            {
                'query_id': q['id'],
                'split': q['split'],
                'question': q['question'],
                'strategy': name,
                'top_k': top_k,
                'budget_chars': budget,
                'filter': filters,
                'gold': q['gold'],
                'results': marked,
                'context': context,
                'context_ids': ids,
                'context_chars': len(context),
                'elapsed_ms': elapsed_ms,
                'hit1': bool(marked and marked[0]['hit']),
                'hit3': any(c['hit'] for c in marked[:3]),
                'context_hit': any(c['hit'] and c['id'] in ids for c in marked),
            }
        )
    summary = {
        'strategy': name,
        'n': len(traces),
        'hit1': sum(t['hit1'] for t in traces),
        'hit3': sum(t['hit3'] for t in traces),
        'context_hit': sum(t['context_hit'] for t in traces),
        'hit1_rate': sum(t['hit1'] for t in traces) / len(traces) if traces else 0,
        'hit3_rate': sum(t['hit3'] for t in traces) / len(traces) if traces else 0,
        'context_evidence_rate': sum(t['context_hit'] for t in traces) / len(traces) if traces else 0,
        'avg_context_chars': statistics.mean(t['context_chars'] for t in traces) if traces else 0,
        'median_ms': statistics.median(t['elapsed_ms'] for t in traces) if traces else 0,
        'top_k': top_k,
        'budget_chars': budget,
        'service_cost_usd': 0,
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Day 7 retrieval benchmark suite')
    parser.add_argument('--chunks', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--chroma', type=Path, default=_STARTER / 'data' / 'chroma')
    parser.add_argument('--strategy', default='C')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--methods',
        default='tfidf,dense,hybrid,hybrid_filter,hybrid_wide',
        help='Comma-separated method names',
    )
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = [
        q for q in json.loads(args.questions.read_text(encoding='utf-8'))
        if q.get('split') == args.split
    ]
    if not questions:
        raise ValueError(f'No questions for split={args.split}')

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    need_dense = any(m in methods for m in ('dense', 'hybrid', 'hybrid_filter', 'hybrid_wide'))
    embedder = MiniLMEmbedder() if need_dense else None
    dense = DenseChromaRetriever(args.chroma, embedder, strategy=args.strategy) if embedder else None
    lexical = TfidfRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical, candidate_k=8) if dense else None
    hybrid_wide = HybridRetriever(dense, lexical, candidate_k=12) if dense else None

    # (search_fn, top_k, budget, use_question_filters)
    configs: dict[str, tuple[Searcher, int, int, bool] | None] = {
        'tfidf': (lexical.search, 3, 800, False),
        'dense': (dense.search, 3, 800, False) if dense else None,
        'hybrid': (hybrid.search, 3, 800, False) if hybrid else None,
        # Improvement 1: apply user-provided section filter on dense side of hybrid
        'hybrid_filter': (hybrid.search, 3, 800, True) if hybrid else None,
        # Improvement 2: wider candidates + larger context budget
        'hybrid_wide': (hybrid_wide.search, 3, 1200, False) if hybrid_wide else None,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for name in methods:
        cfg = configs.get(name)
        if cfg is None:
            raise ValueError(f'Unknown or unavailable method: {name}')
        search_fn, top_k, budget, use_filters = cfg
        print(f'Running {name} on {args.split} (n={len(questions)})...')
        traces, summary = evaluate(
            name, search_fn, questions, top_k=top_k, budget=budget, use_question_filters=use_filters
        )
        summary['split'] = args.split
        out_dir = args.output / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'traces.json').write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        (out_dir / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        all_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    (args.output / 'comparison.json').write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
