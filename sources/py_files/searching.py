from WordPreprocessing import WordPreprocessing
import json
import pandas as pd
import re
from unidecode import unidecode
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Tfidf_prepro import Tfidf_prepro

class Search():   
    def __init__(self, train, data_path, topics_path):
        self.df_docs = None
        self.df_topics = None
        self.df_combined = None
        self.model = None

        self.prep = WordPreprocessing(deaccent=False)
        self.tfidf = Tfidf_prepro()

        if train == True:
            self.load_data(data_path, topics_path)
            self.df_docs.index = self.df_docs['id'].values
            self.clean_data()
            self.preprocess(self.df_docs)
            self.preprocess(self.df_topics)

            self.df_docs.to_csv('semantic-search/BP_data/docs_cleaned.csv')
            self.df_topics.to_csv('semantic-search/BP_data/topics_cleaned.csv')

            self.df_topics = self.df_topics.rename(columns={'description':'text'})
            self.df_combined = self.df_docs.append(self.df_topics, ignore_index=True)
        
            tfidf_df = self.tfidf.calculate_ifidf(self.df_combined)
            self.df_docs = self.tfidf.delete_words(self.df_combined,tfidf_df)

            text = [x for x in self.df_combined['text']]
            title = [x for x in self.df_combined['title']]

            for i in title:
                text.append(i)

            data = []

            for i in text:
                data.append(i.split())

            self.model = Word2Vec(data, vector_size=300, min_count=2, window=5, sg=1, workers=4)
            self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding_w2v(x.split()))
            self.df_topics['vector'] = self.df_topics['text'].apply(lambda x :self.get_embedding_w2v(x.split()))
        
            self.df_docs.to_csv('semantic-search/BP_data/vectorized_data.csv')
            self.df_topics.to_csv('semantic-search/BP_data/vectorized_topics.csv')
            
            self.model.save('semantic-search/BP_data/word2vec_model.model')
        else:
            self.df_docs = pd.read_csv('semantic-search/BP_data/docs_cleaned.csv')
            self.df_topics = pd.read_csv('semantic-search/BP_data/topics_cleaned.csv')

            self.df_topics = self.df_topics.rename(columns={'description':'text'})

            self.model = Word2Vec.load('semantic-search/BP_data/word2vec_model.model')
            self.df_docs['vector'] = self.df_docs['text'].apply(lambda x :self.get_embedding_w2v(x.split()))
            self.df_topics['vector'] = self.df_topics['text'].apply(lambda x :self.get_embedding_w2v(x.split()))

    def process_sentence(self, sentence, prep):
        sent = [self.prep.process_word(w) for w in sentence.split()]
        sent_not_none = ' '.join([str(elem) for elem in sent if len(elem) > 0])
        return sent_not_none

    def load_data(self, path_docs, path_topics):
        with open(path_docs, encoding="utf8") as f:
            docs = json.load(f)

        self.df_docs = pd.DataFrame(docs)

        with open(path_topics, encoding="utf8") as f:
            topics = json.load(f)

        self.df_topics = pd.DataFrame(topics)

        self.df_docs = self.df_docs.drop("date", axis=1)
        self.df_topics = self.df_topics.drop(['narrative', 'lang'], axis=1)

    def clean_text(self, text):
        text=re.sub('\w*\d\w*','', text)
        text=re.sub('\n',' ',text)
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
        text=re.sub('[^A-Za-z0-9]+ ', ' ', text)
        text=re.sub('"().,_', '',text)
        return text

    def clean_df(self, df):
        for col in df.columns:
            if col == 'id':
                continue
            df[col] = df[col].apply(lambda x: unidecode(x, "utf-8"))
            df[col] = df[col].apply(lambda x: self.clean_text(x))

    def clean_data(self):
        self.clean_df(self.df_docs)
        self.clean_df(self.df_topics)

    def preprocess(self, df):
        for col in df.columns:
            if col == 'id':
                continue
            df[col] = df[col].apply(lambda x: self.process_sentence(x, self.prep))

    
    def get_embedding_w2v(self, doc_tokens):
        '''
        Vrátí vektor reprezentujícíc dokument
        '''
        embeddings = []
        if len(doc_tokens)<1:
            return np.zeros(300)
        else:
            for tok in doc_tokens:
                if tok in self.model.wv.key_to_index:
                    embeddings.append(self.model.wv.word_vec(tok))
                else:
                    embeddings.append(np.random.rand(300))
            # mean the vectors of individual words to get the vector of the document
            return np.mean(embeddings, axis=0)


    def ranking_ir(self, query, n):
        '''Vrátí n nejlepších výsledku pro query'''
        
        query = self.process_sentence(query, self.prep)

        vector=self.get_embedding_w2v(query.split())

        documents=self.df_docs[['id','title','text']].copy()
        documents['similarity']=self.df_docs['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
        documents.sort_values(by='similarity',ascending=False,inplace=True)
        
        return documents.head(n).reset_index(drop=True)
    
    