import unicodedata
from regex._regex_core import UNICODE
from spacy.lang.cs import Czech, STOP_WORDS

class WordPreprocessing:
    def __init__(self, lowercase=True, lemmatize=True, remove_stopwords=True, deaccent=True ):
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.deaccent = deaccent

        stopwords_accents = STOP_WORDS
        stopwords_deaccent = [strip_accents(w) for w in stopwords_accents]
        if self.deaccent:
            self.stopwords = stopwords_deaccent
        else:
            self.stopwords = stopwords_accents

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

        if self.remove_stopwords and word in self.stopwords:
            return ''

        return word

    def lemmatize_word(self, word):
        pass
        
def strip_accents(text):
    try:
        text = UNICODE(text, 'utf-8')
    except (TypeError, NameError):
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)