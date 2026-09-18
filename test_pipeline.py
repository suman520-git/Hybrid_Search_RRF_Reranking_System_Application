"""Offline control-flow tests; substitutes do not measure model quality."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from pipeline import search, reciprocal_rank_fusion, tokenize, MODES
from evaluate import metrics


def resources():
    texts = ['password reset', 'company login', 'request limit', 'billing refund']
    docs = [Document(page_content=t, metadata={'id': str(i), 'title': t}) for i,t in enumerate(texts)]
    return {'documents': docs, 'vectors': np.eye(4),
            'embeddings': SimpleNamespace(embed_query=Mock(return_value=[1,0,0,0])),
            'bm25': BM25Okapi([tokenize(t) for t in texts]),
            'reranker': SimpleNamespace(predict=Mock(return_value=[0,1,2,3])),
            'llm': SimpleNamespace(invoke=Mock(return_value=SimpleNamespace(content='Test answer [3]')))}

class Tests(unittest.TestCase):
    def test_rrf_merge_and_arithmetic(self):
        hits=reciprocal_rank_fusion([{'id':'A'}, {'id':'B'}], [{'id':'B'}, {'id':'C'}])
        self.assertEqual([h['id'] for h in hits], ['B','A','C'])
        self.assertAlmostEqual(hits[0]['rrf_score'], 1/62+1/61)
        self.assertEqual(reciprocal_rank_fusion([],[]), [])

    def test_four_modes_and_rerank_cutoff(self):
        for mode in MODES:
            r=resources()
            result=search('password',r,mode,answer=False)
            self.assertLessEqual(len(result['results']),3)
            self.assertEqual(len(result['dense']),0 if mode=='bm25' else 4)
            self.assertEqual(len(result['sparse']),0 if mode=='vector' else 1)
            self.assertEqual(r['reranker'].predict.call_count,int(mode=='hybrid_rerank'))
            self.assertEqual(r['llm'].invoke.call_count,0)
            if mode=='hybrid_rerank':
                self.assertEqual(len(r['reranker'].predict.call_args.args[0]),4)
                self.assertEqual(result['results'][0]['id'],'3')

    def test_empty_bm25_skips_llm(self):
        r=resources()
        result=search('unmatched',r,'bm25')
        self.assertEqual(result['results'],[])
        self.assertIn('not have enough information',result['answer'])
        r['llm'].invoke.assert_not_called()

    def test_answer_and_citations_context(self):
        r=resources()
        self.assertEqual(search('password',r)['answer'],'Test answer [3]')
        self.assertIn('[3]',r['llm'].invoke.call_args.args[0][1][1])

    def test_input_validation(self):
        for q,m in [(' ','vector'),('question','invalid')]:
            with self.assertRaises(ValueError): search(q,resources(),m)

    def test_metrics(self):
        self.assertEqual(metrics(['X','A','Y'],['A','B']),
                         {'hit_at_3':1.0,'recall_at_3':0.5,'mrr_at_3':0.5})

if __name__=='__main__':
    unittest.main()
