import numpy as np
from sklearn.ensemble import RandomForestClassifier


class K_fold_CV(object):
    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = k
        
    def split_K_fold(self):
        self.split_data = [self.X[0+i*len(self.X)//self.k : (i+1)*len(self.X)//self.k] for i in range(self.k)]
        self.split_label = [self.Y[0+i*len(self.Y)//self.k : (i+1)*len(self.Y)//self.k] for i in range(self.k)]

        return None
    
    def get_train_test(self, iter):
        train_data = np.vstack([self.split_data[j] for j in range(self.k) if j!=iter])
        train_label = np.hstack([self.split_label[j] for j in range(self.k) if j!=iter])
        test_data = self.split_data[iter]
        test_label = self.split_label[iter]

        return train_data, train_label, test_data, test_label

    def train_test_step(self, train_data, train_label, test_data, test_label):
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf.fit(train_data, train_label)
        acc = rf.score(test_data, test_label)

        return acc
    
    def fit(self):
        sum_acc = 0
        self.split_K_fold()
        for iter in range(self.k):
            train_data, train_label, test_data, test_label = self.get_train_test(iter)
            acc = self.train_test_step(train_data, train_label, test_data, test_label)
            sum_acc += acc
            print(f"k={iter+1}/{self.k}, acc={acc}")
            
        print(f"average acc={sum_acc / self.k}")

        return None


def word_vector(w2v, tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    
    for word in tokens:
        try:
          vec += w2v.wv[word].reshape((1, size))
          count += 1

        except KeyError:
          continue

    if count != 0:
        vec /= count

    return vec
