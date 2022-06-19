from sources.py_files.word_preprocessing import WordPreprocessing
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.fasttext import load_facebook_model
from sources.py_files.model_search import ModelSearch
import os
from sklearn.feature_extraction.text import TfidfVectorizer


SG = 1 #trénovací algoritmus - 1 pro skip-gram, jinak CBOW
MIN_COUNT = 20 #Minimální frekvence slova aby ho algoritmus použil

class Word2VecSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, model_path:str = None, tfidf_prepro = False, vector_size:int = 300, 
                prepro: WordPreprocessing = WordPreprocessing(), workers:int = 1, column:str = 'title'):
        """Konstruktor

        Args:
            train (bool): zda se má model natrénovat -> True a nebo načíst -> False
            data_path (str): cesta k dokumentům
        """
        super().__init__(train,data_path,seznam_path,save_name,model_path,tfidf_prepro, prepro, workers, column)
        
        self.df_docs = None
        self.seznam_df = None
        self.model = None
        self.vector_size = vector_size

        self.w2v_weights = None
        self.print_settings()

    def print_settings(self):
        super().print_settings()
        print('Nastaveni modelu Word2Vec')
        print(f'tfidf: {self.tfidf_prepro}')

    def start(self):
        if self.seznam_df == None:
            self.seznam_df = self.utils.load_seznam(self.seznam_path)
            self.seznam_df = self.process_documents(self.seznam_df)

        self.get_tfidf_weights()
        if self.train == True:
            self.model_train(self.data_path)
        else:
            self.model_load(self.model_path, self.data_path)
        

    def get_tfidf_weights(self):
        """
        Zjistí váhy jednotlivých slov, váhy se poté berou v potaz kdy se vytváří vektorová reprezentace vět
        """

        if self.tfidf_prepro:
            print('Pocita se tfidf')
            seznam_docs = [x for x in self.seznam_df['doc']]
            tfidf = TfidfVectorizer().fit(seznam_docs)
            self.w2v_weights = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
            print('Tfdif dokonceno')

    def model_train(self, data_path:str):
        #Načtu dokumenty a aplikuji předzpracování
        self.df_docs = self.load_data(data_path)
        self.df_docs = self.process_documents(self.df_docs)

        #Extrahuji texty na kterých se model bude trénovat z dokumentů
        text = [x for x in self.df_docs['text']]
        titles = [x for x in self.df_docs['title']]
        
        #přidám do celkových textů titles
        for t in titles:
            text.append(t)

        seznam_docs = [x for x in self.seznam_df['doc']]
        seznam_titles = [x for x in self.seznam_df['title']]

        for d in seznam_docs:
            text.append(d)

        for t in seznam_titles:
            text.append(t)

        #Jednotlivé texty splitnu na tokeny na kterých se model bude trénovat
        data = [s.split() for s in text]

        print('Trenovani modelu...')
        #Vytvoření a natrénování modelu
        self.model = Word2Vec(data, vector_size=self.vector_size, min_count=MIN_COUNT, sg=SG, workers=self.workers)

        self.df_docs['vector'] = self.df_docs[self.column].apply(lambda x :self.get_embedding(x.split()))
        self.df_docs.to_csv(f'{os.path.dirname(self.data_path)}/docs_cleaned.csv', index=False)
        print('Trenovani dokonceno')
        self.model_save(self.save_name)

    def model_save(self, save_path: str):
        super().model_save(save_path)
        
        self.model.save(save_path + ".model")

    def model_load(self, model_path:str, docs_path:str):
        self.df_docs = self.load_data(docs_path)

        #Podle typu souboru se načte příslušný soubor s vektory
        #Pokud je bin využije se načtený pomocí fasttext
        #INFO na načítaná slova není použit žádný preprocessing
        if '.bin' in model_path:
            print('Nacitani .bin souboru')
            self.model = load_facebook_model(model_path)

        #Pokud se jedná o model načtou se jeho vektory
        elif '.model' in model_path:
            print('Nacitani .model souboru')
            self.model = Word2Vec.load(model_path)

            #Nastavím délku vektoru
            for word in self.model.wv.key_to_index:
                self.vector_size = len(self.model.wv.get_vector(word))
                break

            for word in self.model.wv.key_to_index:
                vector = self.model.wv.get_vector(word)
                self.model.wv.key_to_index[word] = vector
        else:
            print('ERROR: Spatny format nacitaneho modelu, pro w2v model musi byt .bin nebo .model')

        self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding(x.split()))

    def load_data(self, path_docs: str):
        return super().load_data(path_docs)

    def process_documents(self, documents):
        print(f'Cistim data')
        doc = self.utils.clean_df(documents)
        doc = self.utils.preprocess(doc)
        print('Cisteni dokonceno')
        return doc

    def get_embedding(self, doc_tokens):
        embeddings = []
        if len(doc_tokens)<1: #Pokud je málo tokenů vracím nuly
            return np.zeros(self.vector_size)
        else:
            for tok in doc_tokens:
                if self.tfidf_prepro == True and tok not in self.w2v_weights:
                    self.w2v_weights[tok] = 1

                if tok in self.model.wv.key_to_index: #Když se slovo nachází ve slovníku modelu tak ho převedu na na vektor
                    if self.tfidf_prepro == True:
                        embeddings.append(self.model.wv.word_vec(tok) * self.w2v_weights[tok]) #Vážení tfidf
                    else:
                        embeddings.append(self.model.wv.word_vec(tok))
                else: #Když se slovo nenachází ve slovníku tak ho nahradím náhodným vektorem
                    if self.tfidf_prepro == True:
                        embeddings.append(np.random.rand(self.vector_size) * self.w2v_weights[tok]) #Vážení tfidf
                    else:
                        embeddings.append(np.random.rand(self.vector_size)) 

            return np.mean(embeddings, axis=0)

    def ranking_ir(self, query:str, n:int) -> pd.DataFrame:   
        query = self.prepro.process_sentence(query)
        vector=self.get_embedding(query.split())
        
        documents=self.df_docs[['id','title','text']].copy()
        documents['score'] = self.df_docs['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
        documents.sort_values(by='score',ascending=False,inplace=True)
        
        return documents.head(n).reset_index(drop=True)

    