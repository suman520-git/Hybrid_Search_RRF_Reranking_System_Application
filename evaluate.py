"""Run labeled retrieval comparison: python evaluate.py"""
import csv
import json
import os
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pipeline import ROOT, MODES, TOP_K, CANDIDATE_K, RRF_K, load_resources, search


def metrics(predicted, relevant):
    relevant = set(relevant)
    matched = set(predicted) & relevant
    rank = next((i for i, doc_id in enumerate(predicted, 1) if doc_id in relevant), None)
    return {'hit_at_3': float(bool(matched)), 'recall_at_3': len(matched) / len(relevant),
            'mrr_at_3': 1 / rank if rank else 0.0}


def main():
    resources = load_resources()
    questions = json.loads((ROOT / 'data/questions.json').read_text(encoding='utf-8'))
    known = {d.metadata['id'] for d in resources['documents']}
    for row in questions:
        if not row['relevant_ids'] or not set(row['relevant_ids']) <= known:
            raise ValueError('Every evaluation question must have valid relevant document IDs.')
    # Exclude startup and one warm-up per mode from reported latency.
    for mode in MODES:
        search(questions[0]['question'], resources, mode, answer=False)
    details = []
    # Rotate mode order to reduce a fixed ordering advantage.
    for index, row in enumerate(questions):
        modes = MODES[index % 4:] + MODES[:index % 4]
        for mode in modes:
            result = search(row['question'], resources, mode, answer=False)
            predicted = [hit['id'] for hit in result['results']]
            details.append({**row, 'mode': mode, 'predicted_ids': predicted,
                            **metrics(predicted, row['relevant_ids']),
                            'retrieval_ms': result['retrieval_ms'], 'results': result['results']})
    summary = []
    for mode in MODES:
        rows = [r for r in details if r['mode'] == mode]
        summary.append({'mode': mode, **{key: round(sum(r[key] for r in rows) / len(rows), 4)
                         for key in ('hit_at_3', 'recall_at_3', 'mrr_at_3', 'retrieval_ms')}})
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / 'details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    metadata = {'utc': datetime.now(timezone.utc).isoformat(), 'top_k': TOP_K,
                'candidate_k_per_retriever': CANDIDATE_K, 'rrf_k': RRF_K,
                'embedding_model': resources['embeddings'].model,
                'reranker_model': os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L6-v2'),
                'data_sha256': {name: hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest()
                                for name in ('documents.json', 'questions.json')},
                'packages': {p: version(p) for p in ['langchain-core', 'langchain-openai',
                             'sentence-transformers', 'rank-bm25']}}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('mode             Hit@3   Recall@3  MRR@3   mean retrieval ms')
    for row in summary:
        print(f"{row['mode']:16} {row['hit_at_3']:.3f}   {row['recall_at_3']:.3f}     "
              f"{row['mrr_at_3']:.3f}   {row['retrieval_ms']:.2f}")
    best = max(summary, key=lambda row: (row['mrr_at_3'], row['recall_at_3']))
    print(f"Best by MRR@3, then Recall@3: {best['mode']} (ties select first mode).")
    print('Measured results saved in results/. No LLM answer-quality score is computed.')

if __name__ == '__main__':
    main()
