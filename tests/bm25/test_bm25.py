from difflib import SequenceMatcher
import json
from rank_bm25 import BM25Plus
import numpy as np
import os

__thisfilepath__ = os.path.dirname(os.path.abspath(__file__))


def get_string_similar(s1, s2):
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def is_string_similar(s1, s2):
    return get_string_similar(s1, s2) > 0.55


class BM25_Retriever:
    def __init__(self, trueCase=False):
        self.corpus = json.load(
            open(os.path.join(__thisfilepath__, "database.json"))
        )
        self.db = BM25Plus([x if trueCase else x.lower() for x in self.corpus])

    def similar_descriptions(self, text, return_score=False):
        topk = 10
        ans = self.db.get_top_n(text, self.corpus, n=topk)
        if return_score:
            scores = np.sort(self.db.get_scores(text))[::-1][:topk]
            return list(zip(ans, scores))
        return ans


if __name__ == "__main__":
    c = BM25_Retriever(trueCase=True)
    print(c.similar_descriptions("wordA"))
    print(c.similar_descriptions("Keyword"))
