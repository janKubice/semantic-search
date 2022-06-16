from os.path import exists

import pandas as pd
import torch
from sources.py_files.cross_attention_search import CrossAttentionSearch
from sources.py_files.model_search import ModelSearch
from sources.py_files.two_towers_search import TwoTowersSearch
from sources.py_files.word_preprocessing import WordPreprocessing

MAX_LENGTH = 512
BATCH_SIZE = 32
WARMUP_COEF = 0.1
ERROR = -1

class CrossTwoTower(ModelSearch):

    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, validation_path:str,model_path:str = None, tfidf_prepro = False, 
                prepro: WordPreprocessing = WordPreprocessing(), transformer_name = 'paraphrase-multilingual-mpnet-base-v2', 
                workers:int = 1, column:str = 'title'):

        super().__init__(train, data_path, seznam_path, save_name, model_path, tfidf_prepro, prepro, workers, column)
        self.transformer_name = transformer_name
        self.validation_path = validation_path
        self.workers = workers

    def print_settings(self):
        super().print_settings()

    def start(self):
        if self.transformer_name == None:
            print('Nebyl zadan nazev modelu pro kombinovanou metodu')
            exit(ERROR)
            
        if self.train and exists(self.save_name):
            print(f'nazev modelu: {self.save_name}')
            print('Soubor se stejnym jmenem jiz existuje.\nZvolte jiny nazev a spustte program znovu.')
            exit(ERROR)

        if self.train == True and torch.cuda.is_available() == False:
            print('WARNING: Neni aktivni CUDA. Delka trenovani bude znacne delsi!')

        tt_save_name = self.save_name + '_combo_tt'
        ca_save_name = self.save_name + '_combo_ca'

        self.model_tt:TwoTowersSearch = TwoTowersSearch(self.train,
                                        self.data_path, 
                                        self.seznam_path,
                                        tt_save_name,
                                        self.validation_path,
                                        self.model_path, 
                                        self.tfidf_prepro, 
                                        self.prepro,
                                        self.transformer_name,
                                        self.workers)
        self.model_tt.start()

        self.model_ca:CrossAttentionSearch = CrossAttentionSearch(self.train,
                                             self.data_path, 
                                             self.seznam_path,
                                             ca_save_name, 
                                             self.validation_path,
                                             self.model_path, 
                                             self.tfidf_prepro, 
                                             self.prepro,
                                             self.transformer_name,
                                             self.workers)
        self.model_ca.start()
        
    def model_train(self, data_path:str):
        """
        Tato metoda není implementována, vše potřebné se provedlo v konstruktoru a při volání metody start
        Třída volá třídy TwoTowerSearch a CrossAttentionSearch kde je metoda vytvořena
        """
        pass
        
    def model_save(self, save_path:str):
        """
        Tato metoda není implementována, vše potřebné se provedlo v konstruktoru a při volání metody start
        Třída volá třídy TwoTowerSearch a CrossAttentionSearch kde je metoda vytvořena
        """
        pass

    def model_load(self, model_path:str, docs_path):
        """
        Tato metoda není implementována, vše potřebné se provedlo v konstruktoru a při volání metody start
        Třída volá třídy TwoTowerSearch a CrossAttentionSearch kde je metoda vytvořena
        """        
        pass

    def load_data(self, path_docs: str):
        """
        Tato metoda není implementována, vše potřebné se provedlo v konstruktoru a při volání metody start
        Třída volá třídy TwoTowerSearch a CrossAttentionSearch kde je metoda vytvořena
        """        
        pass

    def process_documents(self, documents):
        """
        Tato metoda není implementována, vše potřebné se provedlo v konstruktoru a při volání metody start
        Třída volá třídy TwoTowerSearch a CrossAttentionSearch kde je metoda vytvořena
        """        
        pass

    def get_embedding(self, doc_tokens):
        """
        Tato metoda není implementována, vše potřebné se provedlo v konstruktoru a při volání metody start
        Třída volá třídy TwoTowerSearch a CrossAttentionSearch kde je metoda vytvořena
        """        
        pass

    
    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        tt_result = self.model_tt.ranking_ir(query, n*10)

        #Nastavím výsledek z two tower modelu jako dokumenty pro cross encoder
        self.model_ca.df_docs = tt_result

        ca_result = self.model_ca.ranking_ir(query, n)
        return ca_result
