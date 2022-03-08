from word_preprocessing import WordPreprocessing
import json
import pandas as pd
from unidecode import unidecode
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tfidf_prepro import Tfidf_prepro
from gensim.models.fasttext import load_facebook_model
from model_search import ModelSearch
import utils

class Word2VecSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, model_path:str = None, vector_size:int = 300):
        """Konstruktor

        Args:
            train (bool): zda se má model natrénovat -> True a nebo načíst -> False
            data_path (str): cesta k dokumentům
        """
        self.df_docs = None
        self.model = None
        self.model_path = model_path
        self.vector_size = vector_size

        self.prep = WordPreprocessing(deaccent=False)
        self.tfidf = Tfidf_prepro()

        if train == True:
            self.train(data_path)
        else:
            self.load(model_path, 'semantic-search/BP_data/docs_cleaned.csv')

    def train(self, data_path:str):
        self.df_docs = self.load_data(data_path)
        self.df_docs.index = self.df_docs['id'].values
        utils.clean_df(self.df_docs)
        utils.preprocess(self.df_docs)

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

        self.model = Word2Vec(data, vector_size=self.vector_size, min_count=2, window=5, sg=1, workers=4)
        self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding_w2v(x.split()))
    
        self.df_docs.to_csv('semantic-search/BP_data/vectorized_data.csv')        
        self.model.save('semantic-search/model/word2vec_model.model')

    def save(self, save_path: str):
        return super().save(save_path)

    def load(self, model_path, docs_path):
        self.df_docs = pd.read_csv(docs_path)

        if '.bin' in model_path:
            self.model = load_facebook_model(model_path)
        elif '.model' in model_path:
            self.model = Word2Vec.load(model_path)

            for word in self.model.wv.key_to_index:
                vector = self.model.wv.get_vector(word)

                self.model.wv.key_to_index[self.prep.process_word(word)] = vector
        else:
            print('ERROR: Špatný formát načítaného modelu')

        self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding_w2v(x.split()))

    def load_data(self, path_docs:str) -> pd.DataFrame:
        with open(path_docs, encoding="utf8") as f:
            docs = json.load(f)

        
        df_docs = pd.DataFrame(docs)
        df_docs = df_docs.drop("date", axis=1)
        return df_docs

    def get_embedding_w2v(self, doc_tokens):
        good = 0
        bad = 0
        embeddings = []
        if len(doc_tokens)<1:
            return np.zeros(self.vector_size)
        else:
            for tok in doc_tokens:
                if tok in self.model.wv.key_to_index:
                    embeddings.append(self.model.wv.word_vec(tok))
                    good += 1
                else: 
                    embeddings.append(np.random.rand(self.vector_size)) 
                    bad += 1

            print(good, bad)
            return np.mean(embeddings, axis=0)


    def ranking_ir(self, query:str, n:int) -> pd.DataFrame:   
        query = self.prep.process_sentence(query)

        vector=self.get_embedding_w2v(query.split())

        documents=self.df_docs[['id','title','text']].copy()
        documents['similarity']=self.df_docs['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
        documents.sort_values(by='similarity',ascending=False,inplace=True)
        
        return documents.head(n).reset_index(drop=True)

    