from word_preprocessing import WordPreprocessing
import pandas as pd
from unidecode import unidecode


def clean_df(df:pd.DataFrame, prepro:WordPreprocessing):
    """Vyčiští dataframe

    Args:
        df (pd.DataFrame): dataframe dokuemntu
    """
    for col in df.columns:
        if col == 'id':
            continue
        df[col] = df[col].apply(lambda x: unidecode(x, "utf-8"))
        df[col] = df[col].apply(lambda x: prepro.clean_text(x))

def preprocess(df:pd.DataFrame, prepro:WordPreprocessing):
    """Předzpracuje dokument, metoda je inplace

    Args:
        df (pd.DataFrame): dataframe dokumentů
    """
    for col in df.columns:
        if col == 'id':
            continue
        df[col] = df[col].apply(lambda x: prepro.process_sentence(x))