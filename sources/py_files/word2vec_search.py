from word_preprocessing import WordPreprocessing
import pandas as pd
from unidecode import unidecode
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.fasttext import load_facebook_model
from model_search import ModelSearch
import utils
import os

class Word2VecSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, model_path:str = None, tfidf_prepro = False, vector_size:int = 300, 
                prepro: WordPreprocessing = WordPreprocessing()):
        """Konstruktor

        Args:
            train (bool): zda se má model natrénovat -> True a nebo načíst -> False
            data_path (str): cesta k dokumentům
        """
        super().__init__(train,data_path,seznam_path,save_name,model_path,tfidf_prepro, prepro)
        
        self.df_docs = None
        self.model = None
        self.vector_size = vector_size

        if self.train == True:
            self.model_train(self.data_path)
        else:
            self.model_load(self.model_path, self.data_path)

    def model_train(self, data_path:str):
        super().model_train(data_path)

        #Načtu dokumenty a aplikuji předzpracování
        self.df_docs = self.load_data(data_path)
        self.process_documents(self.df_docs)

        #Extrahuji texty na kterých se model bude trénovat z dokumentů
        text = [x for x in self.df_docs['text']]
        titles = [x for x in self.df_docs['title']]
        
        #přidám do celkových textů titles
        for t in titles:
            text.append(t)

        #Extrahuji texty ze seznam datasetu
        seznam_df = utils.load_seznam(self.seznam_path)
        self.process_documents(seznam_df)

        seznam_docs = [x for x in seznam_df['doc']]
        seznam_titles = [x for x in seznam_df['title']]

        for d in seznam_docs:
            text.append(d)

        for t in seznam_titles:
            text.append(t)

        #Jednotlivé texty splitnu na tokeny na kterých se model bude trénovat
        data = [s.split() for s in text]

        #Vytvoření a natrénování modelu
        self.model = Word2Vec(data, vector_size=self.vector_size, min_count=20, window=5, sg=1, workers=32)

        self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding(x.split()))
        self.df_docs.to_csv(f'{os.path.dirname(self.data_path)}/docs_cleaned.csv', index=False)

        self.model_save(self.save_name)

    def model_save(self, save_path: str):
        super().model_save(save_path)
        
        self.model.save(save_path + ".model")

    def model_load(self, model_path:str, docs_path:str):
        super().model_load(model_path,docs_path)

        self.df_docs = self.load_data(docs_path)

        #Podle typu souboru se načte příslušný soubor s vektory
        #Pokud je bin využije se načtený pomocí fasttext
        if '.bin' in model_path:
            self.model = load_facebook_model(model_path)

        #Pokud se jedná o model načtou se jeho vektory
        elif '.model' in model_path:
            self.model = Word2Vec.load(model_path)

            for word in self.model.wv.key_to_index:
                vector = self.model.wv.get_vector(word)
                self.model.wv.key_to_index[word] = vector
        else:
            print('ERROR: Špatný formát načítaného modelu')

        #self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding(x.split()))

    def load_data(self, path_docs: str):
        return super().load_data(path_docs)

    def process_documents(self, documents):
        super().process_documents(documents)

    def get_embedding(self, doc_tokens):
        embeddings = []
        if len(doc_tokens)<1:
            return np.zeros(self.vector_size)
        else:
            for tok in doc_tokens:
                if tok in self.model.wv.key_to_index: #Když se slovo nachází ve slovníku modelu tak ho převedu na na vektor
                    embeddings.append(self.model.wv.word_vec(tok))
                else: #Když se slovo nenachází ve slovníku tak ho nahradím náhodným vektorem
                    embeddings.append(np.random.rand(self.vector_size)) 

            return np.mean(embeddings, axis=0)

    def ranking_ir(self, query:str, n:int) -> pd.DataFrame:   
        query = self.prepro.process_sentence(query)

        vector=self.get_embedding(query.split())

        #Vytvořím kopii dokumentů, spočítám podobnost, seřadím podle podobnosti a vrátím n nejlepších
        documents=self.df_docs[['id','title','text']].copy()
        documents['similarity'] = self.df_docs['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
        documents.sort_values(by='similarity',ascending=False,inplace=True)
        
        return documents.head(n).reset_index(drop=True)

    