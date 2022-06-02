from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import InputExample
from sources.py_files.word_preprocessing import WordPreprocessing
from sources.py_files.model_search import ModelSearch
from os.path import exists
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

ERROR = -1

class TwoTowersSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, model_path:str = None, tfidf_prepro = False, 
                prepro: WordPreprocessing = WordPreprocessing(), transformer_name = 'paraphrase-multilingual-mpnet-base-v2', workers = 1):
        """Konstruktor

        Args:
            train (bool): zda se má model natrénovat -> True a nebo načíst -> False
            data_path (str): cesta k dokumentům
        """
        super().__init__(train, data_path, seznam_path, save_name, model_path, tfidf_prepro, prepro, workers)
        self.transformer_name = transformer_name

    def print_settings(self):
        print(f'Jmeno transformeru {self.transformer_name}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Model bezi na:', device)
    
    def start(self):
        if exists(self.save_name):
            print(f'nazev modelu pro ulozeni: {self.save_name}')
            print('Soubor se stejnym jmenem jiz existuje.\nZvolte jiny nazev a spustte program znovu.')
            exit()

        if self.train == True and torch.cuda.is_available() == False:
            print('ERROR: Pro trenovani je potreba mit akctivni CUDA.')
            exit()

        self.df_docs = self.load_data(self.data_path)
        self.df_docs = self.process_documents(self.df_docs)

        if self.train == True:
            self.model_train(self.data_path)
        else:
            self.model_load(self.model_path, self.data_path)

        print('Vytvarim embedding pro corpus')
        self.corpus_embeding = self.get_embedding(self.df_docs['title'])
        print('Embedding pro corpus vytvoren')

    def model_train(self, data_path:str):
        if self.transformer_name == None:
            print('Nebyl zadan nazev modelu pro TwoTower')
            exit()

        try:
            self.model = SentenceTransformer(self.transformer_name, device='cuda')
        except BaseException as e:
            print(str(e))
            print(f'Zadene jmeno transformeru neexistuje: {self.transformer_name}')
            exit()

        seznam_df = self.utils.load_seznam(self.seznam_path)
        seznam_df['label'] = seznam_df['label'].astype(float, errors='raise')
        print('Zpracovavani seznam dokumentu')
        seznam_df = self.process_documents(seznam_df)
        print('Zpracovavani seznam dokumentu dokonceno')

        train_examples = []
        for _, row in seznam_df.iterrows():
            train_examples.append(InputExample(texts=[row['query'], row['title']], label=row['label']))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.model)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)], 
                        epochs=100, 
                        warmup_steps=100)

    def model_load(self, model_path: str, docs_path):
        super().model_load(model_path,docs_path)
        self.df_docs = self.load_data(docs_path)
        try:
            self.model = SentenceTransformer(model_path, device='cuda')
        except:
            print('Nepovedlo se nacist CrossEncoder model.')
            exit(ERROR)

    def model_save(self, save_path: str):
        super().model_save(save_path)
        self.model.save(save_path)

    def process_documents(self, documents):
        return super().process_documents(documents)

    def load_data(self, path_docs: str) -> pd.DataFrame:
        return super().load_data(path_docs)

    def get_embedding(self, doc_tokens):
        return self.model.encode(doc_tokens, convert_to_numpy=True, device='cuda' if torch.cuda.is_available() else None, show_progress_bar=True)

    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        documents=self.df_docs[['id','title','text']].copy()

        query_embedding = self.get_embedding(query)
        scores = [cosine_similarity(np.array(query_embedding).reshape(1, -1),np.array(x).reshape(1, -1)).item() for x in self.corpus_embeding]
        documents['score'] = scores
        
        documents.sort_values(by=['score'], ascending=False, inplace=True)
        
        return documents.head(n).reset_index(drop=True)