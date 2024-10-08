from duckduckgo_search import DDGS


class DDGSWrapper:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query, max_results=5):
        return self.ddgs.text(query, max_results=max_results)
