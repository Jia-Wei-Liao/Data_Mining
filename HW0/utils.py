import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class DataPreprocessing():
    def __init__(self, df):
        self.df = df.copy()
    
    def fill_missing_fare(self):
        self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].median())
        
        return None
    
    def fill_missing_age(self):
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        
        return None
    
    def add_sex_code(self):
        self.df['Sex_Code'] = self.df['Sex'].map({'male': 0, 'female': 1}).astype('int')
        
        return None
    
    def add_embarked_code(self):
        self.df['Embarked_Code'] = self.df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).fillna(0).astype('int')
        
        return None
    
    def add_family_size(self):
        self.df['Family_Size'] = self.df['SibSp'] + self.df['Parch'] + 1
        
        return None
    
    def add_alone(self):
        self.df['Alone'] = self.df['Family_Size'].map(lambda x: 0 if x>1 else 1)
        
        return None
    
    def add_adult(self):
        self.df['Adult'] = self.df['Age'].map(lambda x: 1 if x>=18 else 0)
        
        return None
        
    def feature_transform(self):
        self.fill_missing_fare()
        self.fill_missing_age()
        
        self.add_sex_code()
        self.add_embarked_code()
        self.add_family_size()
        self.add_alone()
        self.add_adult()
        
        return None
    
    def get_data(self, train=True):
        self.feature_transform()
        X = self.df[['Pclass', 'Fare', 'Sex_Code', 'Embarked_Code', 'Alone', 'Adult']]
        
        if train: return X, self.df['Survived']
        else: return X


def save_predict(ID, prediction):
    df = pd.DataFrame({'PassengerId': ID, 'Survived': prediction})
    df.to_csv('submission.csv', index=False)
    
    return None
