from model_search import ModelSearch
import pandas as pd
import json

from word2vec_search import Word2VecSearch
from two_towers_search import TwoTowersSearch
from cross_attention_search import CrossAttentionSearch

class Search():
    """
    Třída obsahuje metody na vyhledávání podobných dokumentů
    """

    def __init__(self, train, doc_path, model_path, model_name, vector_size:int = 300) -> None:
        self.train = train
        self.doc_path = doc_path
        self.model_path = model_path
        self.model_name = model_name
        self.vector_size = vector_size

    def get_model(self) -> ModelSearch:
        if self.model_name == 'w2v':
            return self.word2vec()
        elif self.model_name == 'tt':
            return self.two_towers()
        elif self.model_name == 'ca':
            return self.cross_attention()
        else:
            print('Zvolen neexistující model.')
            print('Použije se w2v.')
            self.word2vec()

    def load_queries(self, file:str, model:ModelSearch, top_n:int, result_path:str):
        """Načte dotazy ze souboru, najde top_n nejlepších shod v dokumentech a uloží výsledky do souboru

        Args:
            file (str): cesta k dotazům
            search (Search): instance vyhledávače
            top_n (int): počet nejlepších dokumentů vůči dotazu
            result:path (str): cesta a název souboru kam se uloží výsledky
        """
        results = open(result_path, 'w+')

        with open(file, encoding="utf8") as f:
            queries = json.load(f)

        df_queries = pd.DataFrame(queries)
        #if ['description', 'narrative', 'lang'] in df_queries.columns:
        #    df_queries.drop(['description', 'narrative', 'lang'], axis=1, inplace=True)

        for index,query in df_queries.iterrows():
            top_q = model.ranking_ir(query['title'], top_n)
            for idx, res in top_q.iterrows():
                results.write(f"{query['id']} 0 {res['id']} {idx} {res['similarity']} 0\n")

    def word2vec(self):
        return Word2VecSearch(self.train, self.doc_path, self.model_path, self.vector_size)

    def two_towers(self):
        return TwoTowersSearch()

    def cross_attention(self):
        return CrossAttentionSearch()
        

    

    

    

   

    
    
    
    