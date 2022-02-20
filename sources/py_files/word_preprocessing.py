import unicodedata
from regex._regex_core import UNICODE
from spacy.lang.cs import Czech, STOP_WORDS
import simplemma
import re

class WordPreprocessing:
    """
    Třída na předzpracování slov
    """
    def __init__(self, lowercase=True, lemmatize=True, remove_stopwords=True, deaccent=True, lang='cs'):
        """Konstruktor třídy kde se nastaví parametry předzpracování

        Args:
            lowercase (bool, optional): pokud je True text se převede na lowercase. Defaults to True.
            lemmatize (bool, optional): pokud je True bude na text aplikovaná lematizace. Defaults to True.
            remove_stopwords (bool, optional): pokud je True odstraní se z vět stopová slova. Defaults to True.
            deaccent (bool, optional): pokud je True slova se převedou na základní tvary. Defaults to True.
            lang (str, optional): [description]. Použitý jazyk .Defaults to 'cs'.
        """
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.deaccent = deaccent
        self.langdata = simplemma.load_data(lang)

        self.nlp = Czech()

    def process_word(self, word: str) -> str:
        """Zpracuje jedno slovo a vrátí jeho upravenou podobu, pokud je slovo ve stop slovech vrátí pouze ""

        Args:
            word (str): slovo na zpracování

        Returns:
            str: zpracované slovo
        """

        if self.lowercase:
            word = word.lower()

        if self.lemmatize:
            word = self.lemmatize_word(word)

        if self.deaccent:
            word = self.strip_accents(word)

        if self.remove_stopwords and word in STOP_WORDS:
            return ''

        return word

    def lemmatize_word(self, word: str) -> str:
        """Lematizuje slovo

        Args:
            word (str): slovo na lematizaci

        Returns:
            str: zpracované slovo
        """
        return simplemma.lemmatize(word, self.langdata)
        
    def strip_accents(self, text: str) -> str:
        """Převede slovo do základního tvaru

        Args:
            text (str): slovo na zpracování

        Returns:
            str: upravené slovo
        """
        try:
            text = UNICODE(text, 'utf-8')
        except (TypeError, NameError):
            pass
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return str(text)


    def clean_text(self, text) -> str:
        """Odstraní z textu všechny překážejícíc znaky, webové adresy a podobně

        Args:
            text (str): text ke zpracování

        Returns:
            str: zpracovaný text
        """
        text=re.sub('\w*\d\w*','', text)
        text=re.sub('\n',' ',text)
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
        text=re.sub('[^A-Za-z0-9]+ ', ' ', text)
        text=re.sub('"().,_', '',text)
        return text