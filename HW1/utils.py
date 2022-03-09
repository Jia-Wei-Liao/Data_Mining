import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class DataPreprocessing():
    def __init__(self, df):
        self.df = df.copy()
    
    def fill_missing_data(self):
        self.df['Death Year'] = self.df['Death Year'].fillna(0)
        self.df['Book of Death'] = self.df['Book of Death'].fillna(0)
        self.df['Death Chapter'] = self.df['Death Chapter'].fillna(0)
        self.df['Book Intro Chapter'] = self.df['Book Intro Chapter'].fillna(0)
        
        return None
    
    def get_data(self):
        self.fill_missing_data()
        Y = (self.df['Book of Death']>0).astype(int)
        tmp = self.df.join(pd.get_dummies(self.df["Allegiances"]))
        X = tmp.drop(columns=['Name', 'Allegiances', 'Book Intro Chapter', 'Death Year', 'Book of Death', 'Death Chapter'])
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        
        return None


class GSDecisionTree(DataPreprocessing):
    def __init__(self, data, parameter_space):
        self.data = data
        self.parameter_space = parameter_space
        self.call()
       
    def call(self):
        best_accuracy = 0
        for d, r in self.parameter_space:
            clf = tree.DecisionTreeClassifier(max_depth=d, random_state=r)
            clf = clf.fit(self.data.X_train, self.data.Y_train)
            test_score = clf.score(self.data.X_test, self.data.Y_test)

            if test_score>best_accuracy:
                best_accuracy = test_score
                best_parameter = (d, r)

        self.model = tree.DecisionTreeClassifier(max_depth=best_parameter[0], random_state=best_parameter[1])
        self.model.fit(self.data.X_train, self.data.Y_train)

        print(f"The best (d, r) is {best_parameter} and the best test accuracy is {best_accuracy:.5f}")

        return None
    

class Evaluation():
    def __init__(self, PD, GT):
        self.TN, self.FP, self.FN, self.TP = confusion_matrix(PD, GT).ravel()
        self.precision = self.TP / (self.TP + self.FP)
        self.recall    = self.TP / (self.TP + self.FN)
        self.accuracy = (self.TP + self.TN) / (self.TN + self.FP + self.FN + self.TP)

    def print_confusion_matrix(self):
        print(f"""Confusion matrix:
        [{self.TP}  {self.FP}]
        [{self.FN}  {self.TN}]
        """)

        return None
    

    def print_evaluation(self):
        print(f"Precision: {self.precision:.5f}")
        print(f"Recall:    {self.recall:.5f}")
        print(f"Accuracy:  {self.accuracy:.5f}")

        return None