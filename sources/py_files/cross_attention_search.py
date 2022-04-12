import math
from os.path import exists

import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import \
    CECorrelationEvaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sources.py_files.model_search import ModelSearch
from sources.py_files.word_preprocessing import WordPreprocessing
import sources.py_files.utils

MAX_LENGTH = 512
BATCH_SIZE = 32
WARMUP_COEF = 0.1
WORKERS = 4

class CrossAttentionSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, model_path:str = None, tfidf_prepro = False, 
                prepro: WordPreprocessing = WordPreprocessing()):

        """
            Modely k použití a vyzkoušení:
            : https://huggingface.co/Seznam/small-e-czech #BUG vyhodí runtime chybu s velikostí tenzoru
            : https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
            : 
        """
        super().__init__(train, data_path, seznam_path, save_name, model_path, tfidf_prepro, prepro)

        self.model = CrossEncoder('UWB-AIR/Czert-B-base-cased', num_labels=1, max_length=MAX_LENGTH)
        self.df_docs = None

        if train:
            self.model_train(self.data_path)
        else:
            pass #TODO 
    
    def model_train(self, data_path:str):
        #INFO převzaté odsud https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/cross-encoder/training_stsbenchmark.py
        super().model_train(data_path)

        #if len(self.save_name) > 0 & exists(self.save_name):
         #   print('Soubor se stejným jménem již existuje.\nZvolte jiný název a spusťte program znovu.')
          #  exit()

        self.df_docs = self.load_data(data_path)
        self.process_documents(self.df_docs)

        seznam_df = sources.py_files.utils.load_seznam(self.seznam_path)
        seznam_df['label'] = seznam_df['label'].astype(float, errors='raise')
        self.process_documents(seznam_df)

        samples_train = []

        for _, row in seznam_df.iterrows():
            samples_train.append(InputExample(texts=[row['query'], row['title']], label=row['label']))
            samples_train.append(InputExample(texts=[row['title'], row['query']], label=row['label']))

        train_dataloader = DataLoader(samples_train, shuffle=True, batch_size=BATCH_SIZE)

        num_epochs = 1 
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * WARMUP_COEF)

        self.model.fit(train_dataloader=train_dataloader,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=self.save_name)

    def model_load(self, model_path: str, docs_path:str):
        super().model_load(model_path, docs_path)

        self.df_docs = self.load_data()
        self.model = CrossEncoder(self.model_path)

    def model_save(self, save_path: str):
        super().model_save(save_path)

        self.model.save(save_path)

    def load_data(self, path_docs: str):
        return super().load_data(path_docs)

    def process_documents(self, documents):
        super().process_documents(documents)

    def get_embedding(self, doc_tokens):
        doc = ' '.join(doc_tokens)
        return self.model.encode(doc)

    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        #INFO https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/in_document_search_crossencoder.py 
        # Hledání query v dokumentu

        #INFO https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/cross-encoder/cross-encoder_usage.py
        # Rankování query

        
        ids = [id for id in self.df_docs['id']]
        #Vytvořím všechny dvojice query a documentu, predikuju jejjich scóre vzniklých páru [query, doc]
        model_inputs = [[query, doc] for doc in self.df_docs['text']]
        scores = self.model.predict(model_inputs, num_workers = WORKERS, batch_size=BATCH_SIZE)

        #Seřadím skóre
        results = [{'id':id, 'input': inp, 'score': score} for id, inp, score in zip(ids, model_inputs, scores)]
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        dataframe = pd.DataFrame(results).head(n).reset_index(drop=True)

        return dataframe