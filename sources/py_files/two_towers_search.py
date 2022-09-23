import os
from os.path import exists

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.metrics.pairwise import cosine_similarity
from sources.py_files.model_search import ModelSearch
from sources.py_files.word_preprocessing import WordPreprocessing
from torch.utils.data import DataLoader

ERROR = -1
EPOCHS = 1
WARMUP_STEPS = 500
BATCH_SIZE = 128

class TwoTowersSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, validation_path:str, model_path:str = None, tfidf_prepro = False, 
                prepro: WordPreprocessing = WordPreprocessing(), transformer_name = 'paraphrase-multilingual-mpnet-base-v2', 
                workers = 1, column:str = 'title'):
        """Konstruktor

        Args:
            train (bool): zda se má model natrénovat -> True a nebo načíst -> False
            data_path (str): cesta k dokumentům
        """
        super().__init__(train, data_path, seznam_path, save_name, model_path, tfidf_prepro, prepro, workers, column)
        self.transformer_name = transformer_name
        self.validation_path = validation_path

    def print_settings(self):
        super().print_settings()
        print(f'Jmeno transformeru {self.transformer_name}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Model bezi na:', device)
    
    def start(self):
        if self.train and exists(self.save_name):
            print(f'nazev modelu pro ulozeni: {self.save_name}')
            print('Soubor se stejnym jmenem jiz existuje.\nZvolte jiny nazev pro ulozeni a spustte program znovu.')
            exit()

        if self.train == True and torch.cuda.is_available() == False:
            print('WARNING: Neni aktivni CUDA.')

        self.df_docs = self.load_data(self.data_path)
        self.df_docs = self.process_documents(self.df_docs)

        if self.train == True:
            self.model_train(self.data_path)
        else:
            self.model_load(self.model_path, self.data_path)

        print('Vytvarim embedding pro corpus')
        self.corpus_embeding = self.get_embedding(self.df_docs[self.column])
        print('Embedding pro corpus vytvoren')

    def model_train(self, data_path:str):
        if self.transformer_name == None:
            print('Nebyl zadan nazev modelu pro TwoTower')
            exit()

        try:
            self.model = SentenceTransformer(self.transformer_name, device='cuda' if torch.cuda.is_available() else None)
        except BaseException as e:
            print(str(e))
            print(f'Zadene jmeno transformeru neexistuje: {self.transformer_name}')
            exit()

        seznam_df = self.utils.load_seznam(self.seznam_path)
        seznam_df['label'] = seznam_df['label'].astype(float, errors='raise')
        print('Zpracovavani seznam dokumentu')
        seznam_df = self.process_documents(seznam_df)
        print('Zpracovavani seznam dokumentu dokonceno')

        examples = []
        for _, row in seznam_df.iterrows():
            examples.append(InputExample(texts=[row['query'], row['doc' if self.column == 'text' else self.column]], label=row['label']))



        train_dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)

        train_loss = losses.CosineSimilarityLoss(self.model)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], 
                        epochs=EPOCHS, 
                        warmup_steps=WARMUP_STEPS,
                        show_progress_bar=True,
                        output_path=self.save_name
                        )

        if os.path.exists(self.validation_path):
            seznam_df_val = self.utils.load_seznam(self.validation_path)
            seznam_df_val['label'] = seznam_df_val['label'].astype(float, errors='raise')
            seznam_df_val = self.process_documents(seznam_df_val)

            test_samples = []
            for _, row in seznam_df_val.iterrows():
                test_samples.append(InputExample(texts=[row['query'], row['doc' if self.column == 'text' else self.column]], label=row['label']))

            test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=BATCH_SIZE, name='sts-test')
            test_evaluator(self.model, output_path=self.save_name)

    def model_load(self, model_path: str, docs_path):
        super().model_load(model_path,docs_path)
        self.df_docs = self.load_data(docs_path)
        try:
            self.model = SentenceTransformer(model_path, device='cuda' if torch.cuda.is_available() else None)
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

    def get_embedding(self, doc_tokens, show_progress = True):
        return self.model.encode(doc_tokens, convert_to_numpy=True, device='cuda' if torch.cuda.is_available() else None, show_progress_bar=show_progress, batch_size=BATCH_SIZE)

    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        documents=self.df_docs[['id','title','text']].copy()

        query_embedding = self.get_embedding(query, False)
        scores = [cosine_similarity(np.array(query_embedding).reshape(1, -1),np.array(x).reshape(1, -1)).item() for x in self.corpus_embeding]
        documents['score'] = scores
        
        documents.sort_values(by=['score'], ascending=False, inplace=True)
        
        return documents.head(n).reset_index(drop=True)
