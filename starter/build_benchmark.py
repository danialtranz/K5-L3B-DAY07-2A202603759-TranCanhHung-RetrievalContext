"""Build Day 7 answerable questions + agent cases (with abstain)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from convert_day6 import convert_chunk, convert_question

ABSTAIN_QUESTIONS = [
    {
        'id': 'abs01',
        'split': 'dev',
        'question': 'How does MongoDB choose a shard key for automatic sharding in a replica set?',
        'answerable': False,
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Corpus is Wiley DDBMS Ch.5 concurrency; no MongoDB sharding content.',
    },
    {
        'id': 'abs02',
        'split': 'dev',
        'question': 'What is the default deadlock detection interval in milliseconds for PostgreSQL 16?',
        'answerable': False,
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Chapter discusses locking concepts but not PostgreSQL 16 timer defaults.',
    },
    {
        'id': 'abs03',
        'split': 'test',
        'question': 'How many nodes does a Raft consensus cluster need for majority quorum of 5 voters?',
        'answerable': False,
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Raft consensus is outside the indexed chapter scope.',
    },
    {
        'id': 'abs04',
        'split': 'test',
        'question': 'What is VinUni AI20k lab attendance policy for late submissions?',
        'answerable': False,
        'expected_behavior': 'search_and_abstain',
        'absence_reason': 'Organizational policy is not present in the textbook corpus.',
    },
]


def dehyphen(text: str) -> str:
    return re.sub(r'(\w)-\s+(\w)', r'\1\2', text)


def collapse_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def extract_aligned_span(raw: str, evidence: str) -> str | None:
    """Return a contiguous substring of raw that matches evidence after dehyphen+ws normalize."""
    if evidence in raw:
        return evidence
    target = collapse_ws(dehyphen(evidence))
    raw_n = collapse_ws(dehyphen(raw))
    start_n = raw_n.find(target)
    if start_n < 0:
        return None

    # Walk raw while building collapsed-dehyphen form; record span covering target.
    i = 0
    n_pos = 0
    start_i = None
    end_i = None
    while i < len(raw):
        m = re.match(r'(\w)-\s+(\w)', raw[i:])
        if m:
            piece = m.group(1) + m.group(2)
            for _ in piece:
                if n_pos == start_n and start_i is None:
                    start_i = i
                n_pos += 1
                if n_pos == start_n + len(target):
                    end_i = i + len(m.group(0))
                    return raw[start_i:end_i]
            i += len(m.group(0))
            continue
        ch = raw[i]
        if ch.isspace():
            # collapsed: only emit one space if previous emitted wasn't space and next nonspace exists
            # Simpler: skip all whitespace in normalization counter only when collapsing
            # Find run of whitespace
            j = i
            while j < len(raw) and raw[j].isspace():
                j += 1
            # Does collapsed form need a space here?
            if 0 < n_pos < len(raw_n) and raw_n[n_pos] == ' ':
                if n_pos == start_n and start_i is None:
                    start_i = i
                n_pos += 1
                if n_pos == start_n + len(target):
                    end_i = j
                    return raw[start_i:end_i]
            i = j
            continue
        if n_pos == start_n and start_i is None:
            start_i = i
        n_pos += 1
        i += 1
        if n_pos == start_n + len(target):
            end_i = i
            return raw[start_i:end_i]
    return None


def align_gold(chunks: list[dict], gold: list[dict]) -> list[dict]:
    aligned = []
    for g in gold:
        evidence = g['evidence']
        sid, page = g['source_id'], g['page']
        # Prefer same source+page
        ordered = sorted(
            chunks,
            key=lambda c: (
                0 if c['source_id'] == sid and c['page'] == page else
                1 if c['source_id'] == sid else 2,
                abs(int(c['page']) - int(page)),
            ),
        )
        fixed = None
        for c in ordered:
            span = extract_aligned_span(c['text'], evidence)
            if span:
                fixed = {
                    'source_id': c['source_id'],
                    'page': c['page'],
                    'evidence': span,
                }
                break
        if fixed is None:
            raise ValueError(f'Cannot align evidence for source={sid} page={page}: {evidence[:80]}...')
        aligned.append(fixed)
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-chunks', type=Path, required=True)
    parser.add_argument('--raw-questions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    raw_chunks = json.loads(args.raw_chunks.read_text(encoding='utf-8'))
    raw_questions = json.loads(args.raw_questions.read_text(encoding='utf-8'))
    chunks = [convert_chunk(c) for c in raw_chunks]
    questions = []
    for q in raw_questions:
        converted = convert_question(q)
        if converted.get('filter') is None:
            converted.pop('filter', None)
        converted['gold'] = align_gold(chunks, converted['gold'])
        questions.append(converted)

    cases = []
    for q in questions:
        cases.append(
            {
                'id': q['id'],
                'split': q['split'],
                'question': q['question'],
                'answerable': True,
                'expected_behavior': 'search_and_answer',
            }
        )
    for a in ABSTAIN_QUESTIONS:
        cases.append(
            {
                'id': a['id'],
                'split': a['split'],
                'question': a['question'],
                'answerable': False,
                'expected_behavior': a['expected_behavior'],
                'absence_reason': a['absence_reason'],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'chunks.json').write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (args.output / 'questions.json').write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (args.output / 'cases.json').write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (args.output / 'abstain_questions.json').write_text(
        json.dumps(ABSTAIN_QUESTIONS, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    n_dev = sum(1 for q in questions if q['split'] == 'dev')
    n_test = sum(1 for q in questions if q['split'] == 'test')
    n_abs_dev = sum(1 for a in ABSTAIN_QUESTIONS if a['split'] == 'dev')
    n_abs_test = sum(1 for a in ABSTAIN_QUESTIONS if a['split'] == 'test')
    print(
        f'Wrote {len(chunks)} chunks; answerable {n_dev}dev/{n_test}test; '
        f'abstain {n_abs_dev}dev/{n_abs_test}test -> {args.output}'
    )


if __name__ == '__main__':
    main()
