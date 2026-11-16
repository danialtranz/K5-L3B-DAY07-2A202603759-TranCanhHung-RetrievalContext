"""Export Day 6 SQLite chunks + questions JSON into Day 7-ready intermediates."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path


def export_chunks(db_path: Path, strategy: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT chunk_id, strategy_id, document_id, source, page, text,
               chunk_type, section, figure_id, table_id
        FROM chunks
        WHERE strategy_id = ?
        ORDER BY chunk_id
        """,
        (strategy,),
    ).fetchall()
    conn.close()
    if not rows:
        raise ValueError(f'Không có chunk cho strategy={strategy} trong {db_path}')
    chunks = []
    for r in rows:
        chunks.append(
            {
                'id': r['chunk_id'],
                'chunk_id': r['chunk_id'],
                'document_id': r['document_id'],
                'source_id': r['document_id'],
                'source': r['source'],
                'page': int(r['page']),
                'text': r['text'],
                'metadata': {
                    'strategy_id': r['strategy_id'],
                    'chunk_type': r['chunk_type'] or 'text',
                    'section': r['section'] or '',
                    'figure_id': r['figure_id'] or '',
                    'table_id': r['table_id'] or '',
                },
            }
        )
    return chunks


def load_day6_questions(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get('questions'), list):
        return raw['questions']
    raise ValueError(f'{path}: expect JSON array or object with questions[]')


def copy_runtime_data(day6_data: Path, day7_data: Path) -> None:
    day7_data.mkdir(parents=True, exist_ok=True)
    src_db = day6_data / 'lab.db'
    if not src_db.exists():
        raise FileNotFoundError(f'Missing {src_db}')
    shutil.copy2(src_db, day7_data / 'lab.db')
    src_chroma = day6_data / 'chroma'
    dst_chroma = day7_data / 'chroma'
    if not src_chroma.exists():
        raise FileNotFoundError(f'Missing {src_chroma}')
    if dst_chroma.exists():
        shutil.rmtree(dst_chroma)
    shutil.copytree(src_chroma, dst_chroma)


def main() -> None:
    parser = argparse.ArgumentParser(description='Export Day 6 lab.db / questions for Day 7')
    parser.add_argument(
        '--day6-data',
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / 'K5-L3B-Day06-TranCanhHung-2A202603759-DataIngestionEngineering'
        / 'starter'
        / 'data',
    )
    parser.add_argument('--strategy', default='C')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--copy-runtime', action='store_true',
                        help='Copy lab.db + chroma/ into Day 7 starter/data')
    parser.add_argument(
        '--runtime-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'data',
    )
    args = parser.parse_args()

    db_path = args.day6_data / 'lab.db'
    questions_path = args.day6_data / 'questions.json'
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    if not questions_path.exists():
        raise FileNotFoundError(questions_path)

    args.output.mkdir(parents=True, exist_ok=True)
    chunks = export_chunks(db_path, args.strategy)
    questions = load_day6_questions(questions_path)

    chunks_out = args.output / 'day6_chunks_raw.json'
    questions_out = args.output / 'day6_questions_raw.json'
    chunks_out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    questions_out.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(chunks)} chunks (strategy={args.strategy}) -> {chunks_out}')
    print(f'Exported {len(questions)} questions -> {questions_out}')

    if args.copy_runtime:
        copy_runtime_data(args.day6_data, args.runtime_dir)
        print(f'Copied lab.db + chroma/ -> {args.runtime_dir}')


if __name__ == '__main__':
    main()
