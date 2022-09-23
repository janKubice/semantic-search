import csv
import multiprocessing as mp

import numpy as np
import pandas as pd
from sources.py_files.word_preprocessing import WordPreprocessing

class Utils:
    IGNORE_MASK = ['id', 'label'] #Jaké sloupce sa mají ignorovat při zpracování

    def __init__(self, prepro:WordPreprocessing) -> None:
        self.prepro = prepro

    def print_settings(self):
        """Vypíše nastavení předzpracování"""
        print('Nastaveni predzpracovani:')
        print(f'Deaccent: {str(self.prepro.deaccent)}')
        print(f'Lowercase: {str(self.prepro.lowercase)}')
        print(f'Odstraneni stopwords{str(self.prepro.remove_stopwords)}')
        print(f'Lemmatizace {str(self.prepro.lemmatize)}')

    def clean_df(self, df:pd.DataFrame) -> pd.DataFrame:
        """Vyčiští dataframe

        Args:
            df (pd.DataFrame): dataframe dokuemntu
        """
        df_out = self.parallelize_clean(df, self.parallelize_clean_function)
        return df_out

    def parallelize_clean(self, df:pd.DataFrame, func) -> pd.DataFrame:
        """metoda pro vícevláknové zpracování textu

        Args:
            df (pd.DataFrame): dataframe
            func : funkce pro zpracování samotného textu

        Returns:
            zpracovaný dataframe
        """
        num_processes = mp.cpu_count()
        df_split = np.array_split(df, num_processes)
        with mp.Pool(num_processes) as p:
            df = pd.concat(p.map(func, df_split))
        return df

    def parallelize_clean_function(self, df:pd.DataFrame) -> pd.DataFrame:
        """Metoda zpracuje jednolivé sloupce dataframu

        Args:
            df (pd.DataFrame): dataframe na zpracování
        """
        for col in df.columns:
            if col in self.IGNORE_MASK:
                continue
            df[col] = df[col].apply(lambda x: self.prepro.clean_text(x))
        return df

    def preprocess(self, df:pd.DataFrame) -> pd.DataFrame:
        """Předzpracuje dokument, metoda je inplace a běží ve více vláknech

        Args:
            df (pd.DataFrame): dataframe dokumentů
        """
        df_out = self.parallelize_preprocess(df, self.parallelize_preprocess_function)
        return df_out

    def parallelize_preprocess(self,df:pd.DataFrame, func) -> pd.DataFrame:
        """Vícevláknové zpracování přes předanou funkci

        Args:
            df (pd.DataFrame): dataframe dokumentů
            func : funkce na předzpracování 

        Returns:
            pd.DataFrame: dataframe
        """
        num_processes = mp.cpu_count()
        df_split = np.array_split(df, num_processes)
        with mp.Pool(num_processes) as p:
            df = pd.concat(p.map(func, df_split))
        return df

    def parallelize_preprocess_function(self, df:pd.DataFrame) -> pd.DataFrame:
        """funkce na předzpracování

        Args:
            df (pd.DataFrame): dataframe dokumentů
        """
        for col in df.columns:
            if col in self.IGNORE_MASK:
                continue
            df[col] = df[col].apply(lambda x: self.prepro.process_sentence(x))
        return df

    def load_seznam(self, path:str) -> pd.DataFrame:
        """Načte dokumenty od seznamu

        Args:
            path (str): cesta k seznam datasetu

        Returns:
            pd.DataFrame: seznam dokumenty jako dataframe
        """
        print(f'Nacitam seznam data z: {path}')
        seznam_documents = open(path, encoding="utf8")
        text = csv.reader(seznam_documents, delimiter='\t', quoting=csv.QUOTE_NONE)
        dataframe:pd.DataFrame = pd.DataFrame(text)

        dataframe.columns = dataframe.iloc[0]
        dataframe = dataframe[1:]

        for col in ['query', 'doc', 'title']:
            dataframe = dataframe.loc[(dataframe[col].str.len() != 0)]
            dataframe = dataframe.loc[~dataframe[col].astype(str).str.isnumeric()]

        if len(dataframe) == 0:
            print('Nepovedlo se nacist seznam data')
            exit(-1)

        print('Nacitani seznam dat dokonceno')
        return dataframe
