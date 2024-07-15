import numpy as np
import pandas as pd
from preprocessing import utils_preprocess_text

books_df = pd.read_csv('processed_books_df.csv').fillna("")
training_df = pd.read_csv('training.csv').fillna("")

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()
# train_tfidf = vectorizer.fit_transform(training_df["combined_features"])

# import pickle
# with open("tfidf_vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# with open("train_tfidf.pkl", "wb") as f:
#     pickle.dump(train_tfidf, f)

# train test data split
import numpy as np
from sklearn.model_selection import train_test_split

def train_test_split_tfidf(X, y, test_size=0.2, random_state=42):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_Naive_bayes(X_train, X_test, y_train, y_test):
  classifier = MultinomialNB()
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  return classifier, accuracy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_Random_Forest(X_train, X_test, y_train, y_test):
  classifier = RandomForestClassifier(random_state=42)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  #accuracy_scores.append(accuracy)
  print("Accuracy:", accuracy)
  return classifier, accuracy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_SVC(X_train, X_test, y_train, y_test):
  classifier = SVC(kernel='linear', random_state=42)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  #accuracy_scores.append(accuracy)
  return classifier, accuracy

def filter(preferGenre, preferAuthor, preferRate):
  vectorizer = TfidfVectorizer()
  train_tfidf = vectorizer.fit_transform(training_df["combined_features"])
  # 数据label
  prefer_df = training_df
  preferGenre = utils_preprocess_text(preferGenre)
  preferAuthor = utils_preprocess_text(preferAuthor)
  prefer_df["userPreference"] = prefer_df.apply(lambda row: 1 if(
    (preferGenre in row["genres"] and
    row["average_rating"] >= preferRate)
    or row["author"] in preferAuthor
    ) else 0, axis=1)

  # train model according to userPreference
  X_tr, X_te, y_tr, y_te = train_test_split_tfidf(train_tfidf, prefer_df["userPreference"])
  model_Naive_bayes, acc_NB = train_Naive_bayes(X_tr, X_te, y_tr, y_te)
  model_Random_forest, acc_RF = train_Random_Forest(X_tr, X_te, y_tr, y_te)
  model_SVC, acc_SVC = train_SVC(X_tr, X_te, y_tr, y_te)

  trained_models = {
    model_Naive_bayes:acc_NB,
    model_Random_forest:acc_RF,
    model_SVC:acc_SVC
  }
  best_model = max(trained_models, key=trained_models.get)

  # prediction for full data using the best classfication model
  full_matrix = vectorizer.transform(books_df["combined_features"])
  predicted_preference = best_model.predict(full_matrix)
  print("classification_df_number", np.sum(predicted_preference == 1))
  books_df["predicted_preference"] = predicted_preference

  classifcation_filtered_df = books_df[books_df["predicted_preference"] == 1]

  # filter the data with clustering
  clustering_df = classifcation_filtered_df
  tfidf_clustering = vectorizer.fit_transform(clustering_df['combined_features'])

  import pandas as pd
  from sklearn.cluster import KMeans
  from sklearn.metrics import cohen_kappa_score
  from sklearn.preprocessing import LabelEncoder
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.metrics import silhouette_score

  KMeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
  KMeans.fit(tfidf_clustering)

  clustering_df['cluster_label'] = KMeans.labels_

  cluster = clustering_df.loc[clustering_df['genres'] == preferGenre, 'cluster_label'].mode()[0]
  filtered_df = clustering_df[clustering_df['cluster_label'] == cluster]
  filtered_df[['book_title', 'genres']].drop_duplicates()
  print("clustering_df_number:", filtered_df.shape[0])
  # save data 
  filtered_df.to_csv('filtered_df.csv', index=False)