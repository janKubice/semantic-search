import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sources.py_files.model_search import ModelSearch

class TwoTowersSearch(ModelSearch):

    def __init__(self, train:bool, data_path:str, model_path:str = None):
        """Konstruktor

        Args:
            train (bool): zda se má model natrénovat -> True a nebo načíst -> False
            data_path (str): cesta k dokumentům
        """
        self.df_docs = None
        self.model = None
        self.model_path = model_path

        self.train_dataloader = None
        self.evaluator = None
        self.num_epochs = None
        self.warmup_steps = None
        self.model_save_path = None

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        if train == True:
            self.model_train(data_path)
        else:
            self.model_load(model_path, 'semantic-search/BP_data/docs_cleaned.csv')
    
    def model_train(self, documents: pd.DataFrame):
        self.model.fit(train_dataloader = self.train_dataloader,
                        evaluator = self.evaluator,
                        epochs = self.num_epochs,
                        warmup_steps = self.warmup_steps,
                        output_path = self.model_save_path)

    def model_load(self, model_path: str, docs_path):
        return super().model_load(model_path, docs_path)

    def model_save(self, save_path: str):
        return super().model_save(save_path)

    def load_data(self, path_docs: str) -> pd.DataFrame:
        return super().load_data(path_docs)

    def get_embedding(self, doc_tokens):
        return super().get_embedding(doc_tokens)

    def ranking_ir(self, query: str, n: int) -> pd.DataFrame:
        return super().ranking_ir(query, n)