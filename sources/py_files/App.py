from sources.py_files.searching import Search
import argparse
import sys
from pathlib import Path
import configparser

SUCCESSFUL_END = 0
INPUT_ERROR_END = -1

def read_config(path_to_config) -> list:
    """Přečte config a vrátí ho jako příkazovou řádku která se nastaví argparseru

    Args:
        path_to_config (str): cesta ke konfiguračnímu souboru

    Returns:
        list: příkazová řádka jako list
    """
    path = Path(path_to_config)
    if not path.is_file():
        print('Neplatny konfiguracni soubor - neexistuje.')
        exit(INPUT_ERROR_END)
    if not '.config' in path_to_config:
        print('Neplatny konfigracni soubor - spatna koncovka.')
        exit(INPUT_ERROR_END)

    config = configparser.ConfigParser()
    config.read(path_to_config)

    #Postupně načtu jednotlivé parametry
    doc_path = config.get('PATHS', 'doc_path')
    model_path = config.get('PATHS', 'model_path_load')
    model_path_save = config.get('PATHS', 'model_path_save')
    queries_path = config.get('PATHS', 'queries_path')
    result_path = config.get('PATHS', 'result_file_path')
    seznam_path = config.get('PATHS', 'seznam_path')
    validation_path = config.get('PATHS', 'validation_path')

    model_name = config.get('MODELS', 'model_name')
    transformer_name = config.get('MODELS', 'transformer_model')
    train = config.get('MODELS', 'train')
    save = config.get('MODELS', 'save')
    vector_size = config.get('MODELS', 'vector_size')

    tfidf = config.get('PREPRO', 'tfidf')
    lemma = config.get('PREPRO', 'lemma')
    stopwords = config.get('PREPRO', 'stopwords')
    deaccent = config.get('PREPRO', 'deaccent')
    lang = config.get('PREPRO', 'lang')

    top_n = config.get('SEARCH', 'top_n')
    column = config.get('SEARCH', 'column')

    workers = config.get('OTHERS', 'workers')

    #Vytvořím si string jako z příkazové řádky
    return ['app.py',
            '--train', train,
            '--save', save,
            '--tfidf-prepro', tfidf,
            '--doc-path', doc_path,
            '--model-path', model_path,
            '--model-path-save', model_path_save,
            '--model-name', model_name,
            '--queries-path', queries_path,
            '--top-n', top_n,
            '--result-name', result_path,
            '--vector-size', vector_size,
            '--lemma', lemma,
            '--stopwords', stopwords,
            '--deaccent', deaccent,
            '--lang', lang,
            '--seznam', seznam_path,
            '--transformer-name', transformer_name,
            '--workers', workers,
            '--column', column,
            '--validation_path', validation_path]

def main(train, save, doc_path, model_path, model_name, queries_path, 
        top_n, result_name, vector_size, tfidf_prepro, lemma, stopwords, deaccent, lang, seznam, 
        save_name, transformer_name, workers, column, validation_path):
    """
    Hlavní funkce která se spustí při zapnutí scriptu
    """
    try:
        searcher = Search(eval(train), eval(save), doc_path, model_path, model_name, eval(tfidf_prepro), eval(lemma), 
                        eval(stopwords), eval(deaccent), lang, seznam, save_name, transformer_name, validation_path, int(vector_size), int(workers), column)
    except:
        print('Nektery z parametru nema spravny format.')
        exit(INPUT_ERROR_END)
    
    model = searcher.get_model()
    model.start()

    searcher.load_queries(queries_path, model, int(top_n), result_name)
    exit(SUCCESSFUL_END)

if __name__ == '__main__':   
    #Nadefinování argparser argumentů 
    parser = argparse.ArgumentParser(description='Semantic search')
    parser.add_argument('--config-path', action='store', dest='config', help='Cesta ke konfiguračnímu souboru pro usnadnění práce s programem')
    parser.add_argument('--train', action='store', dest="train", help='Zda se má moddel natrénovat a nebo použít již natrénovaný')
    parser.add_argument('--save', action='store', dest="save", help='Zda se má model uložit', default=True)
    parser.add_argument('--tfidf-prepro', action='store', dest='tfidf_prepro', help='Zda se má využít tfidf při předzpracování')
    parser.add_argument('--doc-path', action="store", dest="doc_path", type=str, help='Cesta s souboru s dokumenty')
    parser.add_argument('--model-path', action='store', dest='model_path', type=str, help="Cesta k již natrénovanému modelu", default=None)
    parser.add_argument('--model-name', action='store', dest='model_name', choices=['w2v', 'tt', 'ca', 'ct'], help="Název modelu který se využije. W2V = word2vec, tt = two towers, ca = cross attention, ct = kombinace ca a tt")
    parser.add_argument('--model-path-save', action='store', dest='save_name', help='Cesta k uložení souboru, bez žádých koncovek, program sám určí koncovku')
    parser.add_argument('--queries-path', action='store', dest='queries_path', help="Cesta k dotazům, využívá se pokud ukládám do souboru", type=str, default=None)
    parser.add_argument('--top-n', action='store', dest='top_n', type=int, help='Počet vrácených nejlepších výsledků', default=100)
    parser.add_argument('--result-name', action='store', dest='result_name', type=str, help='Kam se uloží výsledky k načteným queries', default=None)
    parser.add_argument('--vector-size', action='store', dest='vector_size', type=int, help='Velikost vectoru kterým se bude reprezentovat slovo/dokument', default=300)
    parser.add_argument('--lemma', action='store', dest='lemma', help='Využití lemmatizace při předzpracování')
    parser.add_argument('--stopwords', action='store', dest='stopwords', help='Odstranění stop slov pří předzpracování')
    parser.add_argument('--deaccent', action='store', dest='deaccent', help='Využití deaccentu při předzpracování')
    parser.add_argument('--seznam', action='store', dest='seznam', help='Cesta k dokumentu od seznamu')
    parser.add_argument('--lang', action='store', dest='lang', type=str, help='Jazyk textu')
    parser.add_argument('--transformer-name', action='store', dest='transformer_name', type=str, help='Jaký předtrénovaný model bude využit pro Transformer')
    parser.add_argument('--workers', action='store', dest='workers', type=int, help='Kolik bude potecionálně využito vláken', default=1)
    parser.add_argument('--column', action='store', dest='column', choices=['title', 'text'], help="Jaký sloupec se využije pro trénování a vyhodnocování, například v seznam dokumentech doc=text")
    parser.add_argument('--validation_path', action='store', dest='validation_path', help="Jaký dataset se má využít na validaci, pokud bude zadaná cesta která neexistuje neprovede se validace")


    #INFO Testovací příkazová řádka
    #sys.argv = ['app.py', '--config-path', 'semantic-search/config.config']
    
    if len(sys.argv) < 3:
        print('Musi byt zadany alespon jeden parametr (--config-path)\nHELP\n---------------------------')
        parser.print_help()
        exit(INPUT_ERROR_END)
    
    if '--config-path' in sys.argv:
        print('Nacitani z konfiguracniho souboru')
        try:
            sys.argv = read_config(sys.argv[2])
        except:
            print('Konfiguracni soubor nema spravny format.')
            exit(INPUT_ERROR_END)
    else:
        print('Neplatny prikaz')
        parser.print_help()
        exit(INPUT_ERROR_END)

    arguments = parser.parse_args()
    main(arguments.train, arguments.save, arguments.doc_path, arguments.model_path, arguments.model_name, arguments.queries_path, 
        arguments.top_n, arguments.result_name, arguments.vector_size, arguments.tfidf_prepro, arguments.lemma, arguments.stopwords, 
        arguments.deaccent, arguments.lang, arguments.seznam, arguments.save_name, arguments.transformer_name, arguments.workers, arguments.column, arguments.validation_path)