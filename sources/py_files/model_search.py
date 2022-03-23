from abc import ABC, abstractmethod
import pandas as pd
import json
import csv

class ModelSearch(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def model_train(self, documents:pd.DataFrame):
        """Nastrénuje model

        Args:
            documents (pd.DataFrame): dokumenty na kterých se bude trénovat
        """
        print('Trénování modelu...')
        
    @abstractmethod
    def model_save(self, save_path:str):
        """Uloží natrénovaný model

        Args:
            save_path (str): cesta k uložení (například: C:/modely/muj_model)
        """
        print('Ukládání modelu...')

    @abstractmethod
    def model_load(self, model_path:str, docs_path):
        """
        Načte model
        """
        print('Načítání modelu...')

    @abstractmethod
    def load_data(self, path_docs:str):
        """Načte dokumenty

        Data musí mít formát id,doc,title

        Args:
            path_docs (str): cesta k souboru s dokumenty

        Returns:
            pd.DataFrame: načtené dokumenty
        """
        print('Načítání dat...')

        if '.json' in path_docs:
            with open(path_docs, encoding="utf8") as f:
                docs = json.load(f)
        elif '.tsv' in path_docs:
            with open(path_docs, encoding="utf8") as f:
                docs = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        df_docs = pd.DataFrame(docs)
        self.df_docs = df_docs

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
