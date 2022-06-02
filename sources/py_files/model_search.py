from abc import ABC, abstractmethod
import pandas as pd
import json
import csv

from sources.py_files.utils import Utils
from sources.py_files.word_preprocessing import WordPreprocessing

ERROR = -1

class ModelSearch(ABC):
    """
    Třída slouží jako bázová třída pro ostatní modely
    """
    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, model_path:str = None, tfidf_prepro = False, 
                prepro: WordPreprocessing = WordPreprocessing(), workers = 1):
        super().__init__()
        self.train = train
        self.data_path = data_path
        self.seznam_path = seznam_path
        self.save_name = save_name
        self.model_path = model_path
        self.tfidf_prepro = tfidf_prepro
        self.prepro = prepro
        self.utils:Utils = Utils(prepro)
        self.check_paths()
        self.workers = workers

    def check_paths(self):
        if '.json' not in self.data_path and '.tsv' not in self.data_path :
            print(f'doc_path cesta: {self.data_path}')
            print('doc_path musi byt ve formatu json nebo tsv!')
            exit(ERROR)
        
        if '.' in self.save_name:
            print('model_path_save uvadejte bez koncovky, program sam rozhodne o koncovce.')
            exit(ERROR)

        if '.tsv' not in self.seznam_path:
            print(f'seznam_path cesta: {self.seznam_path}')
            print('seznam_path musi byt formatu .tsv')
            exit(ERROR)

    @abstractmethod
    def print_settings(self):
        pass

    @abstractmethod
    def start(self):
        """
            Dokud se nezavolá tato metoda, model nic nedělá a je pouze nastavený z konstruktoru
        """
        pass

    @abstractmethod
    def model_train(self, data_path:str):
        """Nastrénuje model

        Args:
            data_path (str): cesta k dokumentům na kterých se bude trénovat
        """
        print('Trenovani modelu...')
        
    @abstractmethod
    def model_save(self, save_path:str):
        """Uloží natrénovaný model

        Args:
            save_path (str): cesta k uložení (například: C:/modely/muj_model)
        """
        print(f'Ukladani modelu do {save_path}')

    @abstractmethod
    def model_load(self, model_path:str, docs_path):
        """
        Načte model
        """
        print(f'Nacitani modelu z {model_path}')

    @abstractmethod
    def load_data(self, path_docs:str):
        """Načte dokumenty

        Data musí mít formát id,doc,title

        Args:
            path_docs (str): cesta k souboru s dokumenty

        Returns:
            pd.DataFrame: načtené dokumenty
        """
        print('Nacitani dokumentu...')

        if '.json' in path_docs:
            with open(path_docs, encoding="utf8") as f:
                docs = json.load(f)
        elif '.tsv' in path_docs:
            with open(path_docs, encoding="utf8") as f:
                docs = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        else:
            print('path_docs ERROR: Nepodporovany format dokumentu, musi byt json nebo tsv.')
            print(f'Zadana cesta: {path_docs}')
            exit()

        df_docs = pd.DataFrame(docs)
        df_docs.index = df_docs['id'].values
        return df_docs

    def process_documents(self, documents):
        """Provede zpracování dokumentů 
        """
        doc = self.utils.clean_df(documents)
        doc = self.utils.preprocess(doc)
        return doc

    def get_embedding(self, doc_tokens):
        """Vrátí vektor reprezentujícíc dokument

        Args:
            doc_tokens (array): pole slov dokumentu

        Returns:
            vector: vektor reprezentující dokument
        """
        pass

    def ranking_ir(self, query:str, n:int) -> pd.DataFrame:
        """Nalezne a vrátí n nejlepších výsledků pro query

        Args:
            query (str): dotaz
            n (int): počet nejlepších výsledků

        Returns:
            pd.DataFrame: dataframe s n nejlepšími výsledky
        """
        pass
