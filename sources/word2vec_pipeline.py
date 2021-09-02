import pandas as pd
import json
import re
from unidecode import unidecode
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Function returning vector reperesentation of a document
def get_embedding_w2v(doc_tokens, model):
    """Vrátí vectorovou reprezentaci dokumentu"""
    embeddings = []
    if len(doc_tokens)<1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in model.wv.key_to_index:
                embeddings.append(model.wv.get_vector(tok))
            else:
                embeddings.append(np.random.rand(300))

        return np.mean(embeddings, axis=0)

def clean_text(text):
    """Vyčístí text"""
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    text=re.sub('[^A-Za-z0-9]+ ', ' ', text)
    text=re.sub('"().,_', '',text)
    return text

def lema(text):
    """Provede lematizaci"""
    return text

def rank_query(query, data, model):
    """Ohodnotí dotaz"""
    query = query.lower()
    query = lema(query)
    query = clean_text(query)

    vector = get_embedding_w2v(query.split(), model)

    documents=data[['id','title','text']].copy()
    documents['similarity']=data['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
    documents.sort_values(by='similarity',ascending=False,inplace=True)
  
    return documents.head(10).reset_index(drop=True)

def clean_df(df):
    for col in df.columns:
        if col == 'id':
            continue
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(lambda x: unidecode(x, "utf-8"))
        df[col] = df[col].apply(lambda x: clean_text(x))

def main(model_path, data_path, train):
    
    if train:
        with open("BP_data/czechData.json", encoding="utf8") as f:
            docs = json.load(f)
        df_docs = pd.DataFrame(docs)
        df_docs = df_docs.drop("date", axis=1)

        with open("BP_data/stop_words_czech.json", encoding="utf8") as f:
            stop_words = json.load(f)   

        clean_df(df_docs)
        
        text = [x for x in df_docs['text']]
        title = [x for x in df_docs['title']]

        for i in title:
            text.append(i)

        data = []

        for i in text:
            data.append(i.split())

        model = Word2Vec(data, vector_size=300, min_count=2, window=5, sg=1, workers=4)
        df_docs['vector'] = df_docs['text'].apply(lambda x :get_embedding_w2v(x.split(), model))

        results = rank_query('policie zasahuje proti fanouskum', clean_df, model)
        print(results)

    else:
        data_corpus = pd.read_csv(data_path)
        model = Word2Vec.load(model_path)
    
        results = rank_query('policie zasahuje proti fanouskum', data_corpus, model)
        print(results)



if __name__ == "__main__":
    main('BP_data/w2vmodel.model', 'BP_data/czechData.json', True)

