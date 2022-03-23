from searching import Search
from gui import Gui
import argparse
import sys

def main(train:bool, to_file:bool, doc_path:str, model_path:str, model_name:str, q_path:str, top_n:int, result_file_name:str, vector_size):
    """
    Hlavní funkce která se spustí při zapnutí scriptu
    """
    searcher = Search(train, doc_path, model_path, model_name, vector_size)
    model = searcher.get_model()

    if to_file:
        searcher.load_queries(q_path, model, top_n, result_file_name)
        exit(0)
    else:
        Gui(model, q_path)

if __name__ == '__main__':    
    #INFO nadefinování vstupní hodnot pro testování
    TO_FILE = True #Jestli se mají výsledky uložit do souboru, pokud je False spustí velice jednoduchá aplikace na vyzkoušení funkčnosti
    TRAIN = False #Zda se má W2V natrénovat a nebo použít uložený
    DOCUMENT_PATH = 'X:/Data/BP/czechData_test.json'
    QUERIES_PATH = 'X:/Data/BP/topicData.json'
    MODEL_PATH = 'X:/Data/BP/czech_m.bin'
    RESULT_FILE_NAME = 'X:/Data/BP/results.txt'
    MODEL = 'w2v'
    TOP_N = 100

    command_line = ''

    parser = argparse.ArgumentParser(description='Semantic search')
    parser.add_argument("--train", action="store_true", dest="train")
    parser.add_argument("--save", action="store_true", dest="save")
    parser.add_argument("--doc-path", action="store", dest="doc_path", type=str, help='Cesta s souboru s dokumenty')
    parser.add_argument('--model-path', action='store', dest='model_path', type=str, help="Cesta k již natrénovanému modelu", default=None)
    parser.add_argument('--model-name', action='store', dest='model_name', choices=['w2v', 'tt', 'ca'], help="Název modelu který se využije. W2V = word2vec, tt = two towers, ca = cross attention", required=True)
    parser.add_argument('--queries-path', action='store', dest='queries_path', help="Cesta k dotazům, využívá se pokud ukládám do souboru", type=str, default=None)
    parser.add_argument('--top-n', action='store', dest='top_n', type=int, help='Počet vrácených nejlepších výsledků', default=100)
    parser.add_argument('--result-name', action='store', dest='result_name', type=str, help='Kam se uloží výsledky k načteným queries', default=None)
    parser.add_argument('--vector-size', action='store', dest='vector_size', type=int, help='Velikost vectoru kterým se bude reprezentovat slovo/dokument', default=300)

    #INFO Testovací příkazová řádka
    sys.argv = ['app.py', '--save', '--doc-path',  DOCUMENT_PATH, '--model-name', MODEL, '--queries-path', QUERIES_PATH, '--model-path', MODEL_PATH, '--result-name', RESULT_FILE_NAME]
    
    arguments = parser.parse_args()

    #TODO natrénovat doma W2V, nebo to zkusit nahrát na metacentrum a zkusit trénovat tam
    #TODO zkusit doma _test data (jestli to frčí) a pak na datacentrum celé

    #TODO potom zkusit test na crossEncoder, nejdřív vytvořit test data a pak zase huge data na metacentrum
    main(arguments.train, arguments.save, arguments.doc_path, arguments.model_path, arguments.model_name, arguments.queries_path, arguments.top_n, arguments.result_name, arguments.vector_size)