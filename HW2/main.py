import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import Word2Vec
from utils import *


# Data Preprocessing
df = pd.read_csv('yelp.csv')
data = df[['text', 'stars']]
data['sentence'] = data['text'].map(lambda x: re.sub(r'[0-9\.\-\!\"\(\)\,]', '', x).split())
data['label'] = data['stars'].map(lambda x: int(x>=4))


# TF-IDF
count_vector = CountVectorizer(stop_words='english')
count_vector_matrix = count_vector.fit_transform(data['text']).toarray()
tfidf_transformer = TfidfTransformer()
X1 = tfidf_transformer.fit_transform(count_vector_matrix).toarray()
Y1 = data['label'].tolist()


# Word to Vector
w2v = Word2Vec(data['sentence'], min_count=1, vector_size=250, epochs=1, sg=1)
w2v_matrix = np.zeros((len(data['sentence']), 250)) 

for i in range(len(data['sentence'])):
    w2v_matrix[i,:] = word_vector(w2v, data['sentence'][i], 250)
    
X2 = w2v_matrix
Y2 = data['label'].tolist()


# K-Fold Cross Validation
print("TF-IDF:")
K_fold_CV(X1, Y1, 4).fit()

print("Word to Vector:")
K_fold_CV(X2, Y2, 4).fit()
