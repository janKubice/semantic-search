from word_preprocessing import WordPreprocessing
import json
import pandas as pd
import re
from unidecode import unidecode
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tfidf_prepro import Tfidf_prepro

class Search():
    """
    Třída obsahuje metody na vyhledávání podobných dokumentů
    """

    def __init__(self, train:bool, data_path:str, model_path:str = None):
        """Konstruktor

        Args:
            train (bool): zda se má model natrénovat -> True a nebo načíst -> False
            data_path (str): cesta k dokumentům
        """
        self.df_docs = None
        self.model = None
        self.model_path = model_path

        self.prep = WordPreprocessing(deaccent=False)
        self.tfidf = Tfidf_prepro()

        if train == True:
            self.train(data_path)
        else:
            self.load()

    def load(self):
        """
        Načte model a dokumenty
        """
        self.df_docs = pd.read_csv('semantic-search/BP_data/docs_cleaned.csv')

        self.model = Word2Vec.load(self.model_path)
        self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding_w2v(x.split()))

    def train(self, data_path:str):
        """Natrénuje model word2vec

        Args:
            data_path (str): cesta k dokumentů
        """
        self.df_docs = self.load_data(data_path)
        self.df_docs.index = self.df_docs['id'].values
        self.clean_df(self.df_docs)
        self.preprocess(self.df_docs)

        self.df_docs.to_csv('semantic-search/BP_data/docs_cleaned.csv', index=False)
    
        #TODO problém s ramkou -> potřeba 260gb na tfidf. Předelat ať se to nepočítá celé najednou ale postupně na části.
        #tfidf_df = self.tfidf.calculate_ifidf(self.df_docs)
        #self.df_docs = self.tfidf.delete_words(self.df_docs,tfidf_df)

        text = [x for x in self.df_docs['text']]
        title = [x for x in self.df_docs['title']]

        for i in title:
            text.append(i)

        data = []

        for i in text:
            data.append(i.split())

        self.model = Word2Vec(data, vector_size=300, min_count=2, window=5, sg=1, workers=4)
        self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding_w2v(x.split()))
    
        self.df_docs.to_csv('semantic-search/BP_data/vectorized_data.csv')        
        self.model.save('semantic-search/model/word2vec_model.model')

    def process_sentence(self, sentence):
        """Zpracuje jednu větu dokumentu

        Args:
            sentence (str): věta

        Returns:
            array: pole upravených slov
        """
        sent = [self.prep.process_word(w) for w in sentence.split()]
        sent_not_none = ' '.join([str(elem) for elem in sent if len(elem) > 0])
        return sent_not_none

    def load_data(self, path_docs:str) -> pd.DataFrame:
        """Načte data

        Args:
            path_docs (str): cesta k souboru s dokumenty

        Returns:
            pd.DataFrame: načtené dokumenty
        """
        with open(path_docs, encoding="utf8") as f:
            docs = json.load(f)

        
        df_docs = pd.DataFrame(docs)
        df_docs = df_docs.drop("date", axis=1)
        return df_docs

    def clean_df(self, df):
        """Vyčiští dataframe

        Args:
            df (pd.DataFrame): dataframe dokuemntu
        """
        for col in df.columns:
            if col == 'id':
                continue
            df[col] = df[col].apply(lambda x: unidecode(x, "utf-8"))
            df[col] = df[col].apply(lambda x: self.prep.clean_text(x))

    def preprocess(self, df:pd.DataFrame):
        """Předzpracuje dokument, metoda je inplace

        Args:
            df (pd.DataFrame): dataframe dokumentů
        """
        for col in df.columns:
            if col == 'id':
                continue
            df[col] = df[col].apply(lambda x: self.process_sentence(x))

    
    def get_embedding_w2v(self, doc_tokens):
        """Vrátí vektor reprezentujícíc dokument

        Args:
            doc_tokens (array): pole slov dokumentu

        Returns:
            vector: vektor reprezentující dokument
        """
        embeddings = []
        if len(doc_tokens)<1:
            return np.zeros(300)
        else:
            for tok in doc_tokens:
                if tok in self.model.wv.key_to_index:
                    embeddings.append(self.model.wv.word_vec(tok))
                else:
                    embeddings.append(np.random.rand(300))
            return np.mean(embeddings, axis=0)


    def ranking_ir(self, query:str, n:int) -> pd.DataFrame:
        """Nalezne a vrátí n nejlepších výsledků pro query

        Args:
            query (str): dotaz
            n (int): počet nejlepších výsledků

        Returns:
            pd.DataFrame: dataframe s n nejlepšími výsledky
        """
        
        query = self.process_sentence(query)

        vector=self.get_embedding_w2v(query.split())

        documents=self.df_docs[['id','title','text']].copy()
        documents['similarity']=self.df_docs['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
        documents.sort_values(by='similarity',ascending=False,inplace=True)
        
        return documents.head(n).reset_index(drop=True)
    
    