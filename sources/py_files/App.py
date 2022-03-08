from xgboost import train
from searching import Search
from gui import Gui

def main(train, to_file, doc_path, model_path, model_name, q_path, top_n, result_file_name):
    searcher = Search(train, doc_path, model_path, model_name)
    model = searcher.get_model()

    if to_file:
        searcher.load_queries(q_path, model, top_n, result_file_name)
        exit(0)
    else:
        Gui(model, q_path)

if __name__ == '__main__':    
    TO_FILE = True #Jestli se mají výsledky uložit do souboru, pokud je False spustí velice jednoduchá aplikace na vyzkoušení funkčnosti
    TRAIN = False #Zda se má W2V natrénovat a nebo použít uložený

    DOCUMENT_PATH = 'semantic-search/BP_data/czechData_test.json'
    QUERIES_PATH = 'semantic-search/BP_data/topicData.json'
    MODEL_PATH = 'semantic-search/BP_data/czech_m.bin'
    RESULT_FILE_NAME = 'semantic-search/BP_data/results.txt'
    MODEL = 'w2v'

    TOP_N = 100

    #TODO spuštění z příkazové řádky, použít argparser
    command_line = ''
    main(TRAIN, TO_FILE, DOCUMENT_PATH, MODEL_PATH, MODEL, QUERIES_PATH, TOP_N, RESULT_FILE_NAME)