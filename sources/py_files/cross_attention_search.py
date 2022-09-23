import os
from os.path import exists

import pandas as pd
import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import \
    CECorrelationEvaluator
from sources.py_files.model_search import ModelSearch
from sources.py_files.word_preprocessing import WordPreprocessing
from torch.utils.data import DataLoader

MAX_LENGTH = 512
WARMUP_STEPS = 500
BATCH_SIZE = 128
EPOCHS = 1
ERROR = -1
NUMBER_OF_LABELS = 1 # 1 - kontinuální <0,1>, jinak číslo -> počet labels

class CrossAttentionSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, validation_path:str, model_path:str = None, tfidf_prepro = False, 
                prepro: WordPreprocessing = WordPreprocessing(), transformer_name = 'paraphrase-multilingual-mpnet-base-v2', 
                workers:int = 1, column:str = 'title'):
                
        super().__init__(train, data_path, seznam_path, save_name, model_path, tfidf_prepro, prepro, workers, column)
        self.transformer_name = transformer_name
        self.workers = workers
        self.validation_path = validation_path
        self.print_settings()


    def print_settings(self):
        print(f'Jmeno transformeru: {self.transformer_name} -- CrossAttention')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Model bezi na:', device)
        super().print_settings()

    def start(self):
        self.df_docs = self.load_data(self.data_path)
        self.df_docs = self.process_documents(self.df_docs)

        if self.train:
            self.model_train(self.data_path)
        else:
            self.model_load(self.model_path, self.data_path)
    
    def model_train(self, data_path:str):
        if self.train == True and torch.cuda.is_available() == False:
            print('WARNING: Neni aktivni CUDA. Delka trenovani bude znacne delsi!')

        if exists(self.save_name):
            print(f'nazev modelu: {self.save_name}')
            print('Model se stejnym jmenem jiz existuje.\nZvolte jiny nazev pro ulozeni a spustte program znovu.')
            exit(ERROR)

        if self.transformer_name == None:
            print('Nebyl zadan nazev modelu pro CrossAttention')
            exit(ERROR)
        
        try:
            self.model = CrossEncoder(self.transformer_name, num_labels=NUMBER_OF_LABELS, max_length=MAX_LENGTH, device='cuda' if torch.cuda.is_available() else None)
        except BaseException as e:
            print(str(e))
            print(f'Chyba pri nacitani Transformer modelu: {self.transformer_name}')
            exit(ERROR)

        seznam_df = self.utils.load_seznam(self.seznam_path)
        seznam_df['label'] = seznam_df['label'].astype(float, errors='raise')
        print('Zpracovavani seznam dokumentu')
        seznam_df = self.process_documents(seznam_df)
        print('Zpracovavani seznam dokumentu dokonceno')



        samples_train = []
        for _, row in seznam_df.iterrows():
            samples_train.append(InputExample(texts=[row['query'], row['doc' if self.column == 'text' else self.column]], label=row['label']))
            samples_train.append(InputExample(texts=[row['doc' if self.column == 'text' else self.column], row['query']], label=row['label']))

        train_dataloader = DataLoader(samples_train, shuffle=True, batch_size=BATCH_SIZE)

        self.model.fit(train_dataloader=train_dataloader,
                        epochs=EPOCHS,
                        warmup_steps=WARMUP_STEPS,
                        output_path=self.save_name)

        if os.path.exists(self.validation_path):
            seznam_df_val = self.utils.load_seznam(self.validation_path)
            seznam_df_val['label'] = seznam_df_val['label'].astype(float, errors='raise')
            seznam_df_val = self.process_documents(seznam_df_val)
            
            texts = []
            labels = []
            for _, row in seznam_df_val.iterrows():
                texts.append([row['query'], row['doc' if self.column == 'text' else self.column]])
                labels.append(row['label'])

            test_evaluator = CECorrelationEvaluator(texts,labels, 'results', True)
            test_evaluator(self.model, output_path=self.save_name)

    def model_load(self, model_path: str, docs_path:str):
        super().model_load(model_path,docs_path)
        self.df_docs = self.load_data(docs_path)
        
        try:
            self.model = CrossEncoder(model_path, num_labels=NUMBER_OF_LABELS, max_length=MAX_LENGTH, device='cuda' if torch.cuda.is_available() else None)
        except:
            print('Nepovedlo se nacist CrossEncoder model.')
            exit(ERROR)

    def model_save(self, save_path: str):
        super().model_save(save_path)
        self.model.save(save_path)

    def load_data(self, path_docs: str):
        return super().load_data(path_docs)

    def process_documents(self, documents):
        return super().process_documents(documents)

    def get_embedding(self, doc_tokens):
        doc = ' '.join(doc_tokens)
        return self.model.encode(doc)

    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        ids = [id for id in self.df_docs['id']]

        #Vytvořím všechny dvojice query a documentu, predikuju jejich scóre vzniklých páru [query, doc]
        model_inputs = [[query, doc] for doc in self.df_docs[self.column]]
        scores = self.model.predict(model_inputs, batch_size=BATCH_SIZE, show_progress_bar=True)
        
        #Seřadím skóre
        results = [{'id':id, self.column: inp, 'score': score} for id, inp, score in zip(ids, model_inputs, scores)]
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        dataframe = pd.DataFrame(results).head(n).reset_index(drop=True)

        return dataframe
