import time

from sources.py_files.model_search import ModelSearch
import pandas as pd
import numpy as np
import json

from sources.py_files.word2vec_search import Word2VecSearch
from sources.py_files.two_towers_search import TwoTowersSearch
from sources.py_files.cross_attention_search import CrossAttentionSearch
from sources.py_files.word_preprocessing import WordPreprocessing
from sources.py_files.cross_two_tower import CrossTwoTower

ERROR = -1

class Search():
    """
    Třída obsahuje metody na vyhledávání podobných dokumentů
    """

    def __init__(
        self, train:bool, save:bool, doc_path:str, model_path:str, model_name:str, tfidf_prepro:bool, 
        lemma:bool, remove_stopwords:bool, deaccent:bool, lang:str, seznam:str, save_name:str, transformer_name:str, validation_path:str,
        vector_size:int = 300, workers:int = 1, column:str = 'title') -> None:
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
            transformer_name (str): Jaký Transformer se použije
            validation_path (str): cesta k validačnímu datasetu
            vector_size (int, optional): velikost vektoru reprezentující dokument, využívá se pouze pro w2v. Defaults to 300.
            workers (int, optional): počet vláken
            columns (str, optional): podle jakého sloupce se bude vyhodnocovat podobnost. Pro seznam data text=doc
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
        self.transformer_name = transformer_name
        self.workers = workers
        self.column = column
        self.validation_path = validation_path
        
        self.prepro = WordPreprocessing(True, self.lemma, self.stopwords, self.deaccent, self.lang)

    def get_model(self) -> ModelSearch:
        """Podle volby vrátí model, při nesprávném zadání modelu vrátí defaultní model Word2vec"""
        if self.model_name == 'w2v':
            return self.word2vec()
        elif self.model_name == 'tt':
            return self.two_towers()
        elif self.model_name == 'ca':
            return self.cross_attention()
        elif self.model_name == 'ct':
            return self.cross_two_tower()
        else:
            print('Zvolen neexistujici model.')
            print('Pouzije se w2v.')
            return self.word2vec()

    def load_queries(self, file:str, model:ModelSearch, top_n:int, result_path:str):
        """Načte dotazy ze souboru, najde top_n nejlepších shod v dokumentech a uloží výsledky do souboru

        Args:
            file (str): cesta k dotazům
            search (Search): instance vyhledávače
            top_n (int): počet nejlepších dokumentů vůči dotazu
            result:path (str): cesta a název souboru kam se uloží výsledky
        """
        if '.json' not in file:
            print(f'queries_path cesta: {file}')
            print('queries_path musi byt ve formatu json!')
            exit(ERROR)

        if '.txt' not in result_path:
            print(f'result_path cesta: {result_path}')
            print('result_path musi byt .txt!')
            exit(ERROR)

        print('Nacitani dotazu a hledani relevantnich odpovedi')
        results = open(result_path, 'w+')

        with open(file, encoding="utf8") as f:
            queries = json.load(f)

        df_queries = pd.DataFrame(queries)

        times = []

        for _, query in df_queries.iterrows():
            start = time.time()
            top_q = model.ranking_ir(query['title'], top_n)
            end = time.time()
            times.append(end - start)
            for idx, res in top_q.iterrows():
                results.write(f'{query["id"]} 0 {res["id"]} {idx} {res["score"]} 0\n')

        print(f'Prumerny cas dotazu je: {np.mean(times)} sekundy')

    def word2vec(self):
        """Vrátí model pro word2vec"""
        return Word2VecSearch(self.train, self.doc_path, self.seznam, self.save_name, self.model_path, 
                              self.tfidf_prepro, self.vector_size, self.prepro, self.workers, self.column)

    def two_towers(self):
        """Vrátí model pro two tower"""
        return TwoTowersSearch(self.train, self.doc_path, self.seznam, self.save_name, self.validation_path,self.model_path, 
                               self.tfidf_prepro, self.prepro, self.transformer_name, self.workers, self.column)

    def cross_attention(self):
        """Vrátí model pro cross attention"""
        return CrossAttentionSearch(self.train, self.doc_path, self.seznam, self.save_name, self.validation_path, self.model_path, 
                                    self.tfidf_prepro, self.prepro, self.transformer_name, self.workers, self.column)
        
    def cross_two_tower(self):
        """vrátí kombinovaný model"""
        return CrossTwoTower(self.train, self.doc_path, self.seznam, self.save_name, self.validation_path, self.model_path, 
                             self.tfidf_prepro, self.prepro, self.transformer_name, self.workers, self.column)
    

    

    

   

    
    
    
    