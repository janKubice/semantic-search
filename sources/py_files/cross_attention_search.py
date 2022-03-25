import math

import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import \
    CECorrelationEvaluator
from torch.utils.data import DataLoader

from model_search import ModelSearch
from word_preprocessing import WordPreprocessing



class CrossAttentionSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, model_path:str = None, tfidf_prepro = False, 
                prepro: WordPreprocessing = WordPreprocessing()):

        """
            Modely k použití a vyzkoušení:
            : https://huggingface.co/Seznam/small-e-czech
            : https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
            : a nějaký od favky #QUESTION, zapomněl jsem
        """
        super().__init__(train, data_path, model_path, tfidf_prepro, prepro)

        self.model = CrossEncoder('Seznam/small-e-czech', num_labels=1)
        self.df_docs = None
        self.df_docs = self.load_data(super().data_path)

        if train:
            self.model_train(super().data_path)
    
    def model_train(self, data_path:str):
        train_doc = self.df_docs.sample(frac=0.8) 
        test_doc = self.df_docs.drop(train_doc.index)
        samples_train = []
        samples_test = []

        for index, row in train_doc.iterrows():
            samples_train.append(InputExample(texts=[row('query'), row('title')], label=row('label')))
            samples_train.append(InputExample(texts=[row('title'), row('query')], label=row('label')))

        for index, row in test_doc.iterrows():
            samples_train.append(InputExample(texts=[row('query'), row('title')], label=row('label')))

        train_dataloader = DataLoader(samples_train, shuffle=True, batch_size=16)
        evaluator = CECorrelationEvaluator.from_input_examples(samples_test, name='sts-dev')

        num_epochs = 4
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

        self.model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=super().model_path)

    def model_load(self, model_path: str):
        self.model = CrossEncoder(super().model_path)

    def model_save(self, save_path: str):
        self.model.save(super().model_path)

    def load_data(self, path_docs: str):
        return super().load_data(path_docs)

    def get_embedding(self, doc_tokens):
        doc = ' '.join(doc_tokens)
        return self.model.encode(doc)

    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        #INFO https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/in_document_search_crossencoder.py 
        # Hledání query v dokumentu

        #INFO https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/cross-encoder/cross-encoder_usage.py
        # Rankování query

        #Vytvořím všechny dvojice query a documentu, predikuju jejjich scóre vzniklých páru [query, doc]
        model_inputs = [[query, doc] for doc in self.df_docs['doc']]
        scores = self.model.predict(model_inputs)

        #Seřadím skóre
        results = [{'input': inp, 'score': score} for inp, score in zip(model_inputs, scores)]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        results = results[:n]
        print(results)

        print("Query:", query)
        for hit in results:
            print("Score: {:.2f}".format(hit['score']), "\t", hit['input'][1])

        #TODO vytvořit a vrátit DF