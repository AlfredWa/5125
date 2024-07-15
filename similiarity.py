import pandas as pd
from preprocessing import utils_preprocess_text
from sklearn.neighbors import NearestNeighbors

# books_df = pd.read_csv('filtered_df.csv').fillna("")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()
# train_tfidf = vectorizer.fit_transform(books_df["preprocessed_text"])
# with open("tfidf_vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# with open("train_tfidf.pkl", "rb") as f:
#     X = pickle.load(f)

def recommandbySim(text):
    books_df = pd.read_csv('filtered_df.csv').fillna("")
    processed_text = utils_preprocess_text(text=text, flg_lemm=False, flg_stemm=True)
    vectorizer = TfidfVectorizer()
    train_tfidf = vectorizer.fit_transform(books_df["preprocessed_text"])

    neighbors_model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
    neighbors_model.fit(train_tfidf)
    query_tfidf = vectorizer.transform([processed_text])
    distances, indices = neighbors_model.kneighbors(query_tfidf, n_neighbors=5)
    result_string = ""
    counter = 1
    for dist, idx in zip(distances[0], indices[0]):
        result_string += f"{counter}. {books_df.iloc[idx]['book_title']}\n"
        counter+=1
    return result_string
