import math
from os.path import exists

import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import \
    CECorrelationEvaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model_search import ModelSearch
from word_preprocessing import WordPreprocessing
import utils



class CrossAttentionSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, seznam_path:str, save_name:str, model_path:str = None, tfidf_prepro = False, 
                prepro: WordPreprocessing = WordPreprocessing()):

        """
            Modely k použití a vyzkoušení:
            : https://huggingface.co/Seznam/small-e-czech
            : https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
            : a nějaký od favky #QUESTION, zapomněl jsem
        """
        super().__init__(train, data_path, seznam_path, save_name, model_path, tfidf_prepro, prepro)

        self.model = CrossEncoder('Seznam/small-e-czech', num_labels=1)
        self.df_docs = None

        if train:
            self.model_train(self.data_path)
    
    def model_train(self, data_path:str):
        super().model_train(data_path)

        #if len(self.save_name) > 0 & exists(self.save_name):
         #   print('Soubor se stejným jménem již existuje.\nZvolte jiný název a spusťte program znovu.')
          #  exit()

        self.df_docs = self.load_data(data_path)
        self.process_documents(self.df_docs)

        seznam_df = utils.load_seznam(self.seznam_path)
        seznam_df['label'] = seznam_df['label'].astype(float, errors='raise')
        self.process_documents(seznam_df)

        train_doc = seznam_df.sample(frac=0.8) 
        test_doc = seznam_df.drop(train_doc.index)
        samples_train = []
        samples_test = []

        for _, row in train_doc.iterrows():
            samples_train.append(InputExample(texts=[row['query'], row['title']], label=row['label']))
            samples_train.append(InputExample(texts=[row['title'], row['query']], label=row['label']))

        for _, row in test_doc.iterrows():
            samples_train.append(InputExample(texts=[row['query'], row['title']], label=row['label']))

        train_dataloader = DataLoader(samples_train, shuffle=True, batch_size=32)
        evaluator = CECorrelationEvaluator.from_input_examples(samples_test, name='sts-dev')

        num_epochs = 1 #EDIT 4
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

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

        #Vytvořím všechny dvojice query a documentu, predikuju jejjich scóre vzniklých páru [query, doc]
        ids = [id for id in self.df_docs['id']]
        model_inputs = [[query, doc] for doc in self.df_docs['text']]
        print(model_inputs[:2])
        scores = self.model.predict(model_inputs, num_workers = 4, batch_size=32)

        #Seřadím skóre
        results = [{'input': inp, 'score': score} for inp, score in zip(model_inputs, scores)]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        results = results[:n]
        print(results)

        print("Query:", query)
        for hit in results:
            print("Score: {:.2f}".format(hit['score']), "\t", hit['input'][1])
        exit()
        #TODO vytvořit a vrátit DF