from abc import ABC, abstractmethod
import pandas as pd
class ModelSearch(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, documents:pd.DataFrame):
        """Nastrénuje model

        Args:
            documents (pd.DataFrame): dokumenty na kterých se bude trénovat
        """
        
    @abstractmethod
    def save(self, save_path:str):
        """Uloží natrénovaný model

        Args:
            save_path (str): cesta k uložení (například: C:/modely/)
        """
        pass

    @abstractmethod
    def load(self, model_path:str):
        """
        Načte model
        """
        pass

    @abstractmethod
    def load_data(self, path_docs:str) -> pd.DataFrame:
        """Načte data

        Args:
            path_docs (str): cesta k souboru s dokumenty

        Returns:
            pd.DataFrame: načtené dokumenty
        """
        pass

    def get_embedding_w2v(self, doc_tokens):
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
