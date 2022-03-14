import pandas as pd
from model_search import ModelSearch
from sentence_transformers import CrossEncoder

class CrossAttentionSearch(ModelSearch):

    def __init__(self, docs_path):
        self.model = CrossEncoder()
        self.df_docs = self.load_data(docs_path)
    
    def model_train(self, documents: pd.DataFrame):
        return super().model_train(documents)

    def model_load(self, model_path: str):
        return super().model_load(model_path)

    def model_save(self, save_path: str):
        return super().model_save(save_path)

    def load_data(self, path_docs: str):
        return super().load_data(path_docs)

    def get_embedding_w2v(self, doc_tokens):
        return super().get_embedding_w2v(doc_tokens)

    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        #Vytvořím všechny dvojice query a documentu, predikuju jejjich scóre vzniklých páru [query, doc]
        model_inputs = [[query, passage] for passage in self.df_docs['doc']]
        scores = self.model.predict(model_inputs)

        #Seřadím skóre
        results = [{'input': inp, 'score': score} for inp, score in zip(model_inputs, scores)]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        print(results)

        print("Query:", query)
        for hit in results[:n]:
            print("Score: {:.2f}".format(hit['score']), "\t", hit['input'][1])

        #TODO vytvořit a vrátit DF