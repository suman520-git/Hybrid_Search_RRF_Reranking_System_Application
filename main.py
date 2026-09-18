"""Run a question in one mode or compare all four modes."""
import argparse
import json
from pipeline import MODES, load_resources, search


def main():
    parser = argparse.ArgumentParser(description='Hybrid Search + RRF + Reranking')
    parser.add_argument('question', help='Put your question in quotes.')
    parser.add_argument('--mode', choices=MODES, default='hybrid_rerank')
    parser.add_argument('--compare', action='store_true', help='Run all four modes.')
    parser.add_argument('--no-answer', action='store_true', help='Skip LLM answers.')
    args = parser.parse_args()
    if not args.question.strip():
        parser.error('Question cannot be blank.')
    resources = load_resources()
    for mode in MODES if args.compare else [args.mode]:
        result = search(args.question, resources, mode, answer=not args.no_answer)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
