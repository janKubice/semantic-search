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

        self.prep = WordPreprocessing(deaccent=False, lemmatize=False)
        self.tfidf = Tfidf_prepro()

        if train == True:
            self.model_train(data_path)
        else:
            self.model_load(model_path, 'X:/Data/BP/docs_cleaned.csv')

    def model_train(self, data_path:str):
        super().model_train(data_path)

        self.load_data(data_path)
        self.df_docs.index = self.df_docs['id'].values
        utils.clean_df(self.df_docs)
        utils.preprocess(self.df_docs)

        self.df_docs.to_csv('semantic-search/BP_data/docs_cleaned.csv', index=False)
    
        #TODO využít
        #tfidf_df = self.tfidf.calculate_ifidf(self.df_docs)

        text = [x for x in self.df_docs['text']]
        title = [x for x in self.df_docs['title']]

        for i in title:
            text.append(i)

        data = []

        for i in text:
            data.append(i.split())

        self.model = Word2Vec(data, vector_size=self.vector_size, min_count=2, window=5, sg=1, workers=4)
        self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding(x.split()))
    
        self.df_docs.to_csv('semantic-search/BP_data/vectorized_data.csv')        
        self.model.save('semantic-search/model/word2vec_model.model')

    def model_save(self, save_path: str):
        super().model_save(save_path)
        self.model.save(save_path + ".model")

    def model_load(self, model_path, docs_path):
        #pouze výpis informačního textu
        super().model_load(model_path, docs_path)

        self.df_docs = pd.read_csv(docs_path)

        #Podle typu souboru se načte příslušný soubor
        #Pokud je bin využije se načtený pomocí fasttext
        if '.bin' in model_path:
            self.model = load_facebook_model(model_path)

        #Pokud se jedná o model načtou se vektory
        elif '.model' in model_path:
            self.model = Word2Vec.load(model_path)

            for word in self.model.wv.key_to_index:
                vector = self.model.wv.get_vector(word)

                self.model.wv.key_to_index[self.prep.process_word(word)] = vector
        else:
            print('ERROR: Špatný formát načítaného modelu')

        self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding(x.split()))

    def load_data(self, path_docs: str):
        return super().load_data(path_docs)

    def get_embedding(self, doc_tokens):
        #FIXME self.model.wv.key_to_index obsahuje divnou vocab, zkouknoutjestli metody na prepro fungují jak mají
        good = 0
        bad = 0
        embeddings = []
        lines = []
        if len(doc_tokens)<1:
            return np.zeros(self.vector_size)
        else:
            for tok in doc_tokens:
                if tok in self.model.wv.key_to_index:
                    embeddings.append(self.model.wv.word_vec(tok))
                    good += 1
                else: 
                    embeddings.append(np.random.rand(self.vector_size)) 
                    lines.append(tok)
                    bad += 1

            print(good, bad)
            return np.mean(embeddings, axis=0)


    def ranking_ir(self, query:str, n:int) -> pd.DataFrame:   
        query = self.prep.process_sentence(query)

        vector=self.get_embedding(query.split())

        documents=self.df_docs[['id','title','text']].copy()
        documents['similarity']=self.df_docs['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
        documents.sort_values(by='similarity',ascending=False,inplace=True)
        
        return documents.head(n).reset_index(drop=True)

    