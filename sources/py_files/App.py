from unittest import result
from searching import Search
from gui import Gui
import argparse
import sys
from pathlib import Path
import configparser

def read_config(path_to_config) -> list:
    """Přečte config a vrátí ho jako příkazovou řádku která se nastaví argparseru

    Args:
        path_to_config (str): cesta ke konfiguračnímu souboru

    Returns:
        list: příkazová řádka jako list
    """
    path = Path(path_to_config)
    if not path.is_file():
        print('Neplatný konfigurační soubor.')
        exit()
    if not '.config' in path_to_config:
        print('Neplatný konfigrační soubor.')
        exit()

    config = configparser.ConfigParser()
    config.read(path_to_config)

    doc_path = config.get('PATHS', 'doc_path')
    model_path = config.get('PATHS', 'model_path')
    queries_path = config.get('PATHS', 'queries_path')
    result_path = config.get('PATHS', 'result_file_path')

    model_name = config.get('MODELS', 'model_name')
    train = bool(config.get('MODELS', 'train'))
    save = bool(config.get('MODELS', 'save'))
    vector_size = int(config.get('MODELS', 'vector_size'))

    tfidf = bool(config.get('PREPRO', 'tfidf'))
    lemma = bool(config.get('PREPRO', 'lemma'))
    stopwords = bool(config.get('PREPRO', 'stopwords'))
    deaccent = bool(config.get('PREPRO', 'deaccent'))
    lang = config.get('PREPRO', 'lang')

    top_n = int(config.get('SEARCH', 'top_n'))
    to_file = bool(config.get('SEARCH', 'to_file'))

    return ['app.py',
            '--train', train,
            '--save', save,
            '--to-file', to_file,
            '--tfidf-prepro', tfidf,
            '--doc-path', doc_path,
            '--model-path', model_path,
            '--model-name', model_name,
            '--queries-path', queries_path,
            '--top-n', top_n,
            '--result-name', result_path,
            '--vector-size', vector_size,
            '--lemma', lemma,
            '--stopwords', stopwords,
            '--deaccent', deaccent,
            '--lang', lang]

def main(train, to_file, save, doc_path, model_path, model_name, queries_path, 
        top_n, result_name, vector_size, tfidf_prepro, lemma, stopwords, deaccent, lang):
    """
    Hlavní funkce která se spustí při zapnutí scriptu
    """
    searcher = Search(train, save, doc_path, model_path, model_name, tfidf_prepro, lemma, stopwords, deaccent, lang, vector_size)
    model = searcher.get_model()

    if to_file:
        searcher.load_queries(queries_path, model, top_n, result_name)
        exit(0)
    else:
        Gui(model, doc_path)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Semantic search')
    parser.add_argument('--config-path', action='store', dest='config', help='Cesta ke konfiguračnímu souboru pro usnadnění práce s programem')
    parser.add_argument("--train", action="store", dest="train", help='Zda se má moddel natrénovat a nebo použít již natrénovaný')
    parser.add_argument("--save", action="store", dest="save", help='Zda se má model uložit', default=True)
    parser.add_argument('--to-file', action='store', dest='to_file', help='Zda se výsledky vyhledávání mají uložit do souboru')
    parser.add_argument('--tfidf-prepro', action='store', dest='tfidf_prepro', help='Zda se má využít tfidf při předzpracování')
    parser.add_argument("--doc-path", action="store", dest="doc_path", type=str, help='Cesta s souboru s dokumenty')
    parser.add_argument('--model-path', action='store', dest='model_path', type=str, help="Cesta k již natrénovanému modelu", default=None)
    parser.add_argument('--model-name', action='store', dest='model_name', choices=['w2v', 'tt', 'ca'], help="Název modelu který se využije. W2V = word2vec, tt = two towers, ca = cross attention")
    parser.add_argument('--queries-path', action='store', dest='queries_path', help="Cesta k dotazům, využívá se pokud ukládám do souboru", type=str, default=None)
    parser.add_argument('--top-n', action='store', dest='top_n', type=int, help='Počet vrácených nejlepších výsledků', default=100)
    parser.add_argument('--result-name', action='store', dest='result_name', type=str, help='Kam se uloží výsledky k načteným queries', default=None)
    parser.add_argument('--vector-size', action='store', dest='vector_size', type=int, help='Velikost vectoru kterým se bude reprezentovat slovo/dokument', default=300)
    parser.add_argument('--lemma', action='store', dest='lemma', help='Využití lemmatizace při předzpracování')
    parser.add_argument('--stopwords', action='store', dest='stopwords', help='Odstranění stop slov pří předzpracování')
    parser.add_argument('--deaccent', action='store', dest='deaccent', help='Využití deaccentu při předzpracování')
    parser.add_argument('--lang', action='store', dest='lang', type=str, help='Jazyk textu')

    #INFO Testovací příkazová řádka
    sys.argv = ['app.py', '--config-path', 'semantic-search/config.config']
    
    if len(sys.argv) < 3:
        print('Musí být zadaný alespoň jeden parametr\nHELP\n---------------------------')
        parser.print_help()
    
    if '--config-path' in sys.argv:
        print('Načítání z konfiguračního souboru')
        sys.argv = read_config(sys.argv[2])

    arguments = parser.parse_args()
    #TODO natrénovat doma W2V, nebo to zkusit nahrát na metacentrum a zkusit trénovat tam
    #TODO zkusit doma _test data (jestli to frčí) a pak na datacentrum celé
    #TODO potom zkusit test na crossEncoder, nejdřív vytvořit test data a pak zase huge data na metacentrum
    main(arguments.train, arguments.to_file, arguments.save, arguments.doc_path, arguments.model_path, arguments.model_name, arguments.queries_path, 
        arguments.top_n, arguments.result_name, arguments.vector_size, arguments.tfidf_prepro, arguments.lemma, arguments.stopwords, arguments.deccent, arguments.lang)