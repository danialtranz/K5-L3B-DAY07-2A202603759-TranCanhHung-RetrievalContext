"""Offline grounded controller used when no live LLM key is available.

This still goes through run_agent + real retrieval. Prefer OpenAICompatController
when OPENAI_API_KEY / OPENROUTER_API_KEY works.
"""
from __future__ import annotations

import re
from typing import Any


_OOD_PATTERNS = (
    r'\bmongodb\b',
    r'\braft\b',
    r'\bvinuni\b',
    r'\bpostgresql\s*16\b',
    r'\battendance policy\b',
)


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r'[a-z0-9]+', text.casefold()) if len(t) > 2}


def _parse_context_blocks(context: str) -> dict[str, str]:
    """Map chunk-id -> full text body from pack_context output."""
    blocks: dict[str, str] = {}
    current_id: str | None = None
    body_lines: list[str] = []
    header_re = re.compile(r'^\[([^\]|]+) \|')
    for line in context.splitlines():
        m = header_re.match(line)
        if m:
            if current_id is not None:
                blocks[current_id] = '\n'.join(body_lines).strip()
            current_id = m.group(1).strip()
            body_lines = []
            continue
        if current_id is not None:
            body_lines.append(line)
    if current_id is not None:
        blocks[current_id] = '\n'.join(body_lines).strip()
    return blocks


def _pick_chunk(question: str, context_ids: list[str], blocks: dict[str, str]) -> str | None:
    q_terms = _tokenize(question)
    best_id = None
    best_score = -1.0
    for cid in context_ids:
        text = blocks.get(cid) or ''
        if not text:
            continue
        t_terms = _tokenize(text)
        overlap = len(q_terms & t_terms)
        score = overlap / max(1, len(q_terms))
        if score > best_score:
            best_score = score
            best_id = cid
    return best_id or (context_ids[0] if context_ids else None)


class ExtractiveController:
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        q = question.strip()
        if not history:
            return {'action': 'search', 'query': q}

        context = history[-1].get('context') or ''
        context_ids = list(history[-1].get('context_ids') or [])
        blocks = _parse_context_blocks(context)
        if not context_ids or not any(blocks.get(i) for i in context_ids):
            return {
                'action': 'answer',
                'answer': 'Không đủ bằng chứng để trả lời.',
                'citations': [],
            }

        lowered = q.casefold()
        if any(re.search(p, lowered) for p in _OOD_PATTERNS):
            return {
                'action': 'answer',
                'answer': 'Không đủ bằng chứng để trả lời.',
                'citations': [],
            }

        cid = _pick_chunk(q, context_ids, blocks)
        if not cid:
            return {
                'action': 'answer',
                'answer': 'Không đủ bằng chứng để trả lời.',
                'citations': [],
            }
        body = re.sub(r'\s+', ' ', blocks.get(cid) or '').strip()
        # Prefer a sentence window containing overlapping keywords when possible.
        q_terms = _tokenize(q)
        sentences = re.split(r'(?<=[.!?])\s+', body)
        chosen = body
        best = -1
        for s in sentences:
            score = len(q_terms & _tokenize(s))
            if score > best:
                best = score
                chosen = s
        snippet = chosen[:420].strip()
        if not snippet:
            return {
                'action': 'answer',
                'answer': 'Không đủ bằng chứng để trả lời.',
                'citations': [],
            }
        answer = f'{snippet} [{cid}]'
        return {'action': 'answer', 'answer': answer, 'citations': [cid]}
