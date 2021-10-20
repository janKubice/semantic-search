import unicodedata
from regex._regex_core import UNICODE
from spacy.lang.cs import Czech, STOP_WORDS
import simplemma

class WordPreprocessing:
    def __init__(self, lowercase=True, lemmatize=True, remove_stopwords=True, deaccent=True, lang='cs'):
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.deaccent = deaccent
        self.langdata = simplemma.load_data(lang)

        self.nlp = Czech()

    def process_word(self, word):
        if type(word) != str:
            return ''

        if self.lowercase:
            word = word.lower()

        if self.lemmatize:
            word = self.lemmatize_word(word)

        if self.deaccent:
            word = strip_accents(word)

        if self.remove_stopwords and word in STOP_WORDS:
            return ''

        return word

    def lemmatize_word(self, word):
        return simplemma.lemmatize(word, self.langdata)
        
def strip_accents(text):
    try:
        text = UNICODE(text, 'utf-8')
    except (TypeError, NameError):
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)