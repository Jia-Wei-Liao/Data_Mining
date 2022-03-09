import pandas as pd
import graphviz
from sklearn import tree
from utils import *


# Load data
df = pd.read_csv('character-deaths.csv')
data = DataPreprocessing(df)
data.get_data()


# Grid Search
max_depth_space = range(1, 6)
random_state_space = range(100)
parameter_space = [(d, r) for d in max_depth_space for r in random_state_space]
GSDT = GSDecisionTree(data, parameter_space)


# Predicted
data.X_train_predict = GSDT.model.predict(data.X_train)
data.X_test_predict = GSDT.model.predict(data.X_test)
evtrain = Evaluation(data.X_train_predict, data.Y_train)
evtest = Evaluation(data.X_test_predict, data.Y_test)


# Evaluate
print('==================')
print('Training:')
evtrain.print_confusion_matrix()
evtrain.print_evaluation()

print('==================')
print('Test:')
evtest.print_confusion_matrix()
evtest.print_evaluation()


# Plot the Decision Tree
dot_data = tree.export_graphviz(
    decision_tree=GSDT.model,
    out_file=None,
    feature_names=data.X_train.columns.values,
    class_names=data.Y_train.name,
    filled=True,
    rounded=True,
    special_characters=True)

graph=graphviz.Source(dot_data)
graph.render('decision_tree', view=True, format='jpg')

