"""OpenAI-compatible tool-calling controller for Day 7 agent loop."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """You are a grounded RAG agent over a distributed databases textbook chapter.
You operate inside an agent loop: decide to search or give a final answer.
Rules:
- The first search already used the user's question. Only search again if a concrete missing fact is absent from context.
- Keep follow-up queries short and close to the user wording. Do not stuff synonym lists that drift retrieval.
- If context already states the needed fact (even briefly, e.g. naming ACID as atomicity, consistency, isolation, and durability), call final_answer with status=answered.
- When answering, every factual claim must be supported by retrieved chunks.
- Put citation markers like [chunk-id] exactly in the answer text and list the same IDs in citations.
- Abstain only when context truly lacks the answer after useful searches — not because definitions feel incomplete.
- Never invent chunk IDs. Only cite IDs present in the provided context.
- Do not use gold answers or hidden labels; rely only on retrieved context.
"""

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search',
            'description': 'Search the document corpus for evidence relevant to a query.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query focused on needed evidence.',
                    }
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'final_answer',
            'description': 'Return a grounded answer with citations, or abstain.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'status': {
                        'type': 'string',
                        'enum': ['answered', 'abstain'],
                    },
                    'answer': {
                        'type': 'string',
                        'description': 'Final answer text; include [chunk-id] markers when answering.',
                    },
                    'citations': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Chunk IDs that support the answer; empty when abstaining.',
                    },
                },
                'required': ['status', 'answer', 'citations'],
            },
        },
    },
]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Prefer Day 7 starter/.env, fall back to cwd.
    starter_env = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(starter_env):
        load_dotenv(starter_env, override=False)
    load_dotenv(override=False)


def build_client() -> tuple[OpenAI, str]:
    _load_dotenv()
    openrouter = os.getenv('OPENROUTER_API_KEY', '').strip()
    openai_key = os.getenv('OPENAI_API_KEY', '').strip()
    model = os.getenv('CHAT_MODEL', '').strip()
    if openrouter:
        client = OpenAI(api_key=openrouter, base_url='https://openrouter.ai/api/v1')
        return client, model or 'openai/gpt-4o-mini'
    if openai_key:
        client = OpenAI(api_key=openai_key)
        return client, model or 'gpt-4o-mini'
    raise RuntimeError('Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env')


class OpenAICompatController:
    """Controller.decide adapter expected by starter/agentic.py."""

    def __init__(self, client: OpenAI | None = None, model: str | None = None) -> None:
        if client is None or model is None:
            client, model = build_client()
        self.client = client
        self.model = model

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        # First turn: search with the raw user question so lexical/hybrid
        # ranking is not poisoned by LLM synonym stuffing.
        if not history:
            q = question.strip()
            if not q:
                raise ValueError('question must be non-empty')
            return {'action': 'search', 'query': q}

        messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'Question: {question}'},
        ]
        latest = history[-1]
        context = latest.get('context') or ''
        context_ids = latest.get('context_ids') or []
        queries = latest.get('queries') or []
        messages.append(
            {
                'role': 'user',
                'content': (
                    f'Previous search queries: {queries}\n'
                    f'Available context chunk IDs: {context_ids}\n'
                    f'Context:\n{context}\n\n'
                    'Prefer final_answer if the context already supports the question. '
                    'Search again only for a missing fact; otherwise answer or abstain.'
                ),
            }
        )

        # Some reasoning models (e.g. gpt-6-luna) reject tools unless effort is none.
        create_kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'tools': TOOLS,
            'tool_choice': 'required',
            'temperature': 0,
        }
        effort = os.getenv('REASONING_EFFORT', 'none').strip()
        if effort:
            create_kwargs['reasoning_effort'] = effort
        try:
            response = self.client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            # Retry once without reasoning_effort for models that reject the field.
            if 'reasoning_effort' in create_kwargs and 'reasoning_effort' in str(exc):
                create_kwargs.pop('reasoning_effort', None)
                response = self.client.chat.completions.create(**create_kwargs)
            else:
                raise
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            # Fallback: try parse plain JSON answer
            return self._parse_content(msg.content or '')

        call = tool_calls[0]
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or '{}')
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid tool arguments from model: {exc}') from exc

        if name == 'search':
            query = args.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query must be non-empty string')
            return {'action': 'search', 'query': query.strip()}
        if name == 'final_answer':
            status = args.get('status')
            answer = args.get('answer')
            citations = args.get('citations') or []
            if status == 'abstain':
                return {
                    'action': 'answer',
                    'answer': answer if isinstance(answer, str) and answer.strip() else 'Không đủ bằng chứng để trả lời.',
                    'citations': [],
                }
            if not isinstance(answer, str) or not isinstance(citations, list):
                raise ValueError('final_answer payload invalid')
            citations = [c for c in citations if isinstance(c, str)]
            # Ensure markers exist if citations provided
            for cid in citations:
                if f'[{cid}]' not in answer:
                    answer = f'{answer.rstrip()} [{cid}]'
            return {'action': 'answer', 'answer': answer, 'citations': citations}
        raise ValueError(f'Unknown tool: {name}')

    def _parse_content(self, content: str) -> dict[str, Any]:
        match = re.search(r'\{.*\}', content, flags=re.S)
        if not match:
            return {'action': 'answer', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': []}
        data = json.loads(match.group(0))
        if data.get('action') == 'search':
            return {'action': 'search', 'query': str(data.get('query') or '').strip()}
        return {
            'action': 'answer',
            'answer': str(data.get('answer') or 'Không đủ bằng chứng để trả lời.'),
            'citations': list(data.get('citations') or []),
        }
