from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec, KeyedVectors   
import os

import nltk
nltk.download('punkt')

pathSave = "Saved/rews"
pathData = "Datas/reviews_data.txt"

def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                print("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)

if __name__ == '__main__':

    if os.path.isfile(pathSave):
        model = Word2Vec.load(pathSave)
        print("Model loaded")
        
        w1 = "police"
        print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1))

        w1 = ["room"]
        print("Most similar to {0}".format(w1),model.wv.most_similar(positive=w1,topn=6))
    else:
        documents = list(read_input(pathData))
        print("Done reading data file")

        model = gensim.models.Word2Vec(
            documents,
            vector_size=150,
            window=10,
            min_count=2,
            workers=10)
        
        model.train(documents, total_examples=len(documents), epochs=10)
        
        model.save(pathSave)
        print("Model saved")
