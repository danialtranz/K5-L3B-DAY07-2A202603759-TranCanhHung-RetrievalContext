"""Run the Day 7 agent loop on cases.json with a real model + Day 6 retriever."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_STARTER = Path(__file__).resolve().parent
if str(_STARTER) not in sys.path:
    sys.path.insert(0, str(_STARTER))

from agentic import run_agent  # noqa: E402
from extractive_controller import ExtractiveController  # noqa: E402
from llm_controller import OpenAICompatController, build_client  # noqa: E402
from retrieval.chroma_store import DenseChromaRetriever  # noqa: E402
from retrieval.embeddings import MiniLMEmbedder  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402
from retrieval.tfidf import TfidfRetriever  # noqa: E402


def build_retriever(name: str, chunks: list[dict], chroma: Path, strategy: str):
    lexical = TfidfRetriever(chunks)
    if name == 'tfidf':
        return lexical
    embedder = MiniLMEmbedder()
    dense = DenseChromaRetriever(chroma, embedder, strategy=strategy)
    if name == 'dense':
        return dense
    if name in ('hybrid', 'hybrid_filter', 'hybrid_wide'):
        candidate_k = 12 if name == 'hybrid_wide' else 8
        return HybridRetriever(dense, lexical, candidate_k=candidate_k)
    raise ValueError(f'Unknown retriever: {name}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run agentic RAG benchmark')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--chunks', type=Path, required=True)
    parser.add_argument('--questions', type=Path, help='Optional; used for filters on answerable items')
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--retriever', default='hybrid_wide')
    parser.add_argument('--chroma', type=Path, default=_STARTER / 'data' / 'chroma')
    parser.add_argument('--strategy', default='C')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=1200)
    parser.add_argument('--limit', type=int, default=0, help='Optional cap for smoke tests')
    parser.add_argument('--ids', default='', help='Comma-separated case IDs to run')
    parser.add_argument(
        '--controller',
        choices=('auto', 'llm', 'extractive'),
        default='auto',
        help='auto: LLM if key works else extractive offline controller',
    )
    args = parser.parse_args()

    cases = [
        c for c in json.loads(args.cases.read_text(encoding='utf-8'))
        if c.get('split') == args.split
    ]
    if args.ids.strip():
        wanted = {x.strip() for x in args.ids.split(',') if x.strip()}
        cases = [c for c in cases if c['id'] in wanted]
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError('No cases selected')

    filters_by_id: dict[str, Any] = {}
    if args.questions and args.questions.exists():
        for q in json.loads(args.questions.read_text(encoding='utf-8')):
            if q.get('filter'):
                filters_by_id[q['id']] = q['filter']

    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    retriever = build_retriever(args.retriever, chunks, args.chroma, args.strategy)
    use_filters = args.retriever == 'hybrid_filter'

    controller_name = args.controller
    if args.controller == 'llm':
        controller = OpenAICompatController()
        controller_name = 'llm'
    elif args.controller == 'extractive':
        controller = ExtractiveController()
        controller_name = 'extractive'
    else:
        try:
            client, model = build_client()
            # Probe one tiny call is expensive; trust key presence and catch runtime errors.
            controller = OpenAICompatController(client=client, model=model)
            controller_name = 'llm'
        except Exception as exc:  # noqa: BLE001
            print(f'LLM unavailable ({exc}); falling back to extractive controller')
            controller = ExtractiveController()
            controller_name = 'extractive'

    traces = []
    args.output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        cid = case['id']
        question = case['question']
        filters = filters_by_id.get(cid) if use_filters else None
        started = time.perf_counter()
        try:
            result = run_agent(
                question,
                controller,
                retriever,
                filters=filters,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            trace = {
                'query_id': cid,
                'split': case['split'],
                'question': question,
                'answerable': case.get('answerable'),
                'status': result['status'],
                'answer': result.get('answer'),
                'citations': result.get('citations') or [],
                'events': result.get('events') or [],
                'elapsed_ms': (time.perf_counter() - started) * 1000,
                'retriever': args.retriever,
                'controller': controller_name,
                'model': os.getenv('CHAT_MODEL') or controller_name,
            }
        except Exception as exc:  # noqa: BLE001 - record provider/retriever failures
            # If LLM auth fails mid-run and controller was llm, retry once with extractive.
            if controller_name == 'llm' and '401' in str(exc):
                print(f'{cid}: LLM auth failed; switching to extractive for remaining cases')
                controller = ExtractiveController()
                controller_name = 'extractive'
                try:
                    result = run_agent(
                        question,
                        controller,
                        retriever,
                        filters=filters,
                        max_searches=args.max_searches,
                        top_k=args.top_k,
                        budget=args.budget,
                    )
                    trace = {
                        'query_id': cid,
                        'split': case['split'],
                        'question': question,
                        'answerable': case.get('answerable'),
                        'status': result['status'],
                        'answer': result.get('answer'),
                        'citations': result.get('citations') or [],
                        'events': result.get('events') or [],
                        'elapsed_ms': (time.perf_counter() - started) * 1000,
                        'retriever': args.retriever,
                        'controller': controller_name,
                        'model': controller_name,
                    }
                    traces.append(trace)
                    print(f"{cid}: {trace['status']}")
                    continue
                except Exception as exc2:  # noqa: BLE001
                    exc = exc2
            trace = {
                'query_id': cid,
                'split': case['split'],
                'question': question,
                'answerable': case.get('answerable'),
                'status': 'error',
                'answer': None,
                'citations': [],
                'events': [{'event': 'error', 'error': str(exc)}],
                'elapsed_ms': (time.perf_counter() - started) * 1000,
                'retriever': args.retriever,
                'controller': controller_name,
                'model': os.getenv('CHAT_MODEL') or controller_name,
            }
        traces.append(trace)
        print(f"{cid}: {trace['status']}")

    (args.output / 'traces.json').write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    summary = {
        'split': args.split,
        'retriever': args.retriever,
        'controller': controller_name,
        'n': len(traces),
        'answered': sum(t['status'] == 'answered' for t in traces),
        'abstain': sum(t['status'] == 'abstain' for t in traces),
        'error': sum(t['status'] == 'error' for t in traces),
        'median_ms': sorted(t['elapsed_ms'] for t in traces)[len(traces) // 2] if traces else 0,
    }
    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
