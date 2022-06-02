import csv
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sources.py_files.word_preprocessing import WordPreprocessing



class Utils:
    IGNORE_MASK = ['id', 'label'] #Jaké sloupce sa mají ignorovat při zpracování

    def __init__(self, prepro:WordPreprocessing) -> None:
        self.prepro = prepro

    def clean_df(self, df:pd.DataFrame):
        """Vyčiští dataframe

        Args:
            df (pd.DataFrame): dataframe dokuemntu
        """

        """for col in df.columns:
            if col in IGNORE_MASK:
                continue
            df[col] = df[col].apply(lambda x: prepro.clean_text(x))"""

        df_out = self.parallelize_clean(df, self.parallelize_clean_function)
        return df_out

    def parallelize_clean(self,df, func):
        num_processes = mp.cpu_count()
        df_split = np.array_split(df, num_processes)
        with mp.Pool(num_processes) as p:
            df = pd.concat(p.map(func, df_split))
        return df

    def parallelize_clean_function(self, df):
        for col in df.columns:
            if col in self.IGNORE_MASK:
                continue
            df[col] = df[col].apply(lambda x: self.prepro.clean_text(x))
        return df

    def preprocess(self, df:pd.DataFrame):
        """Předzpracuje dokument, metoda je inplace

        Args:
            df (pd.DataFrame): dataframe dokumentů
        """
        """for col in df.columns:
            if col in self.IGNORE_MASK:
                continue
            df[col] = df[col].apply(lambda x: self.prepro.process_sentence(x))"""
        df_out = self.parallelize_preprocess(df, self.parallelize_preprocess_function)

        return df_out

    def parallelize_preprocess(self,df, func):
        num_processes = mp.cpu_count()
        df_split = np.array_split(df, num_processes)
        with mp.Pool(num_processes) as p:
            df = pd.concat(p.map(func, df_split))
        return df

    def parallelize_preprocess_function(self, df):
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
        dataframe = pd.DataFrame(text)

        dataframe.columns = dataframe.iloc[0]
        dataframe = dataframe[1:]

        if len(dataframe) == 0:
            print('Nepovedlo se nacist seznam data')
            exit(-1)

        print('Nacitani seznam dat dokonceno')
        return dataframe
