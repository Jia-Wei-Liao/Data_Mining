import pandas as pd
from utils import *


train_n = 700

train_df = pd.read_csv('train.csv')
trdp = DataPreprocessing(train_df)
X, Y = trdp.get_data()

train_X, train_Y = X[:train_n], Y[:train_n]
valid_X, valid_Y = X[train_n:], Y[train_n:]

model = RandomForestClassifier(
    random_state=2,
    n_estimators=250,
    min_samples_split=20,
    oob_score=True)

model.fit(train_X, train_Y)
model.oob_score_
model.score(train_X, train_Y)
model.score(valid_X, valid_Y)

test_df = pd.read_csv('test.csv')
tsdp    = DataPreprocessing(test_df)
test_X  = tsdp.get_data(train=False)
test_pred = model.predict(test_X)
save_predict(test_df.PassengerId, test_pred)