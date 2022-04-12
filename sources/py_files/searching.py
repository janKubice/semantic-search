from sources.py_files.model_search import ModelSearch
import pandas as pd
import json

from sources.py_files.word2vec_search import Word2VecSearch
from sources.py_files.two_towers_search import TwoTowersSearch
from sources.py_files.cross_attention_search import CrossAttentionSearch
from sources.py_files.word_preprocessing import WordPreprocessing

class Search():
    """
    Třída obsahuje metody na vyhledávání podobných dokumentů
    """

    def __init__(
        self, train:bool, save:bool, doc_path:str, model_path:str, model_name:str, tfidf_prepro:bool, 
        lemma:bool, remove_stopwords:bool, deaccent:bool, lang:str, seznam:str, save_name:str, vector_size:int = 300) -> None:
        """Vytvoří objekt pro vyhledávání se zadaným nastavením

        Args:
            train (bool): zda se má model natrénovat a nebo použít již natrénovaný
            doc_path (str): cesta k dokumentům ve kterých se bude vyhledávat
            model_path (str): cesta k uložení modelu a nebo k načtení modelu, záleží na volbě *train*
            model_name (str): jméno modelu který se má použít
            tfidf_prepro (bool): volba zda se má využít tfidf v preprocesingu
            lemma (bool): Zda se při předzpracování má použít lemmatizace
            remove_stopwords (bool): Zda se při předzpracování mají odstranit stop slova
            deaccent (bool): Zda se při předzpracování má odstranit akcent
            lang (str): Využitý jazyk při předzpracování
            seznam (str): Cesta k seznam dokumentu
            save_name (str): Cesta i s názvem modelu bez koncovky, program sám určí koncovku
            vector_size (int, optional): velikost vektoru reprezentující dokument, využívá se pouze pro w2v. Defaults to 300.
        """
        
        self.train = train
        self.save = save
        self.doc_path = doc_path
        self.model_path = model_path
        self.model_name = model_name
        self.vector_size = vector_size
        self.tfidf_prepro = tfidf_prepro
        self.lemma = lemma
        self.stopwords = remove_stopwords
        self.deaccent = deaccent
        self.lang = lang
        self.seznam = seznam
        self.save_name = save_name

        self.prepro = WordPreprocessing(True, self.lemma, self.stopwords, self.deaccent, self.lang)

    def get_model(self) -> ModelSearch:
        """Podle volby vrátí model, při nesprávném zadání modelu vrátí defaultní model Word2vec"""
        if self.model_name == 'w2v':
            return self.word2vec()
        elif self.model_name == 'tt':
            return self.two_towers()
        elif self.model_name == 'ca':
            return self.cross_attention()
        else:
            print('Zvolen neexistující model.')
            print('Použije se w2v.')
            return self.word2vec()

    def load_queries(self, file:str, model:ModelSearch, top_n:int, result_path:str):
        """Načte dotazy ze souboru, najde top_n nejlepších shod v dokumentech a uloží výsledky do souboru

        Args:
            file (str): cesta k dotazům
            search (Search): instance vyhledávače
            top_n (int): počet nejlepších dokumentů vůči dotazu
            result:path (str): cesta a název souboru kam se uloží výsledky
        """
        print('Načítání dotazů a hledání relevantních odpovědí')
        results = open(result_path, 'w+')

        with open(file, encoding="utf8") as f:
            queries = json.load(f)

        df_queries = pd.DataFrame(queries)

        for _, query in df_queries.iterrows():
            top_q = model.ranking_ir(query['title'], top_n)
            for idx, res in top_q.iterrows():
                results.write(f"{query['id']} 0 {res['id']} {idx} {res['score']} 0\n")

    def word2vec(self):
        """Vrátí model pro word2vec"""
        return Word2VecSearch(self.train, self.doc_path, self.seznam, self.save_name, self.model_path, self.tfidf_prepro, self.vector_size, self.prepro)

    def two_towers(self):
        """Vrátí model pro two tower"""
        return TwoTowersSearch()

    def cross_attention(self):
        """Vráít model pro cross attention"""
        return CrossAttentionSearch(self.train, self.doc_path, self.seznam, self.save_name, self.model_path, self.tfidf_prepro, self.prepro)
        

    

    

    

   

    
    
    
    