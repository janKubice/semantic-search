from tkinter.constants import WORD
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re

class Tfidf_prepro:
    """
    Trída pro TF-IDF zpracování textu
    """

    def calculate_ifidf(self, docs):
        """Pro každý dokument spočítá pro jednotlivá slova jejich váhy

        Args:
            docs (array): Pole dokumentů, každý prvek v poli je text jednoho dokumentu

        Returns:
            pd.dataframe: dataframe kde řádky jsou dokumenty a sloupce jsou slova, 
            každý prvek v tabulce obsahuje hodnotu td-idf pro dané slovo v daném dokumentu
        """
        sents = docs.text.values
        cv = CountVectorizer()
        word_count_vector = cv.fit_transform(sents)

        tfidf_transformer = TfidfTransformer()
        X = tfidf_transformer.fit_transform(word_count_vector)

        tf_idf = pd.DataFrame(X.toarray() ,columns=cv.get_feature_names())
        tf_idf.index = list(docs['id'])
        return tf_idf

    def delete_words(self, docs ,tf_idf):
        """Odstraní z dokumentů slova, která pro daný dokument nejsou podstatná

        Args:
            docs (array): pole dokumentů
            tf_idf (pd.dataframe): vypočítané tf-idf pro dokumenty

        Returns:
            array: upravené pole dokumentů
        """

        characters_to_remove = "\"'"
        for row in docs.iterrows():
            row_id = row[1][1]
            row_text = row[1][0]

            pattern = "[" + characters_to_remove + "]"
            row_text = re.sub(pattern, "", row_text)
            row_text = row_text.split(' ')

            intersection_set = set. intersection(set(row_text), set(tf_idf))
            new_sent = ' '.join([word for word in intersection_set if np.mean(tf_idf[word]) > tf_idf.loc[row_id][word]])
            docs['text'][row_id] = new_sent

        return docs
            
                
