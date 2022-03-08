import pandas as pd
from model_search import ModelSearch

class TwoTowersSearch(ModelSearch):
    
    def train(self, documents: pd.DataFrame):
        return super().train(documents)

    def load(self, model_path: str):
        return super().load(model_path)

    def save(self, save_path: str):
        return super().save(save_path)

    def load_data(self, path_docs: str) -> pd.DataFrame:
        return super().load_data(path_docs)

    def get_embedding_w2v(self, doc_tokens):
        return super().get_embedding_w2v(doc_tokens)

    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        return super().ranking_ir(query, n)