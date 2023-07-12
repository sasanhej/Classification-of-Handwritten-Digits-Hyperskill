import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from sklearn.model_selection import GridSearchCV

simplefilter(action='ignore', category=FutureWarning)

(X_train, y_train) = tf.keras.datasets.mnist.load_data(path="mnist.npz")[0]
X_train = X_train.reshape(-1, 784)
'''
#Stage 1/5
print(f"Classes: {np.unique(y_train)}")
print(f"Features' shape: {X_train.shape}")
print(f"Target's shape: {y_train.shape}")
print(f"min: {X_train.min()}, max: {X_train.max()}")
'''
#Stage 2/5
X_train, X_test, y_train, y_test = train_test_split(X_train[:6000], y_train[:6000], test_size=0.3, random_state=40)
'''
print(f"x_train shape: {X_train.shape}")
print(f"x_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print('Proportion of samples per class in train set:')
print(pd.Series(y_train).value_counts(normalize=True))
'''
#Stage 3/5:
def fit_predict_eval(model, features_train=X_train, features_test=X_test, target_train=y_train, target_test=y_test, random_state=40):
    model.fit(features_train, target_train)
    target_prediction = model.predict(features_test)
    score = np.mean(target_prediction == target_test)
    print(f'Model: {model}\nAccuracy: {score}\n')

'''
fit_predict_eval(KNeighborsClassifier())
fit_predict_eval(DecisionTreeClassifier(random_state=40))
fit_predict_eval(LogisticRegression(random_state=40))
fit_predict_eval(RandomForestClassifier(random_state=40))
print(f'The answer to the question: {"RandomForestClassifier"} - {0.939}')
'''
#Stage 4/5:
X_train_norm = Normalizer().fit_transform(X_train)
X_test_norm = Normalizer().fit_transform(X_test)

def fit_predict_eval_norm(model, features_train=X_train_norm, features_test=X_test_norm, target_train=y_train, target_test=y_test, random_state=40):
    model.fit(features_train, target_train)
    target_prediction = model.predict(features_test)
    score = accuracy_score(target_test, target_prediction)
    #print(f'Model: {model}\nAccuracy: {score}\n')

fit_predict_eval_norm(KNeighborsClassifier())
fit_predict_eval_norm(DecisionTreeClassifier(random_state=40))
fit_predict_eval_norm(LogisticRegression(random_state=40))
fit_predict_eval_norm(RandomForestClassifier(random_state=40))
#print(f'The answer to the 1st question: yes')
#print(f'The answer to the 2nd question: {"KNeighborsClassifier"}-{0.953}, {"RandomForestClassifier"}-{0.937}')

#stage 5/5
random_state = 40
models = [KNeighborsClassifier(), RandomForestClassifier(random_state=random_state)]
param = [{'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']},
         {'n_estimators': [200, 300], 'max_features': ['auto', 'log2'], 'class_weight': ['balanced', 'balanced_subsample']}]

optkn = GridSearchCV(estimator=models[0], param_grid=param[0], scoring='accuracy', n_jobs=-1)
optkn.fit(X_train_norm, y_train)
optkn = optkn.best_estimator_
print(f'K-nearest neighbours algorithm\nbest estimator: {optkn}')
preds = optkn.predict(X_test_norm)
print(f'accuracy: {round(accuracy_score(preds, y_test), 3)}\n')

optrf = GridSearchCV(estimator=models[1], param_grid=param[1], scoring='accuracy', n_jobs=-1)
optrf.fit(X_train_norm, y_train)
optrf = optrf.best_estimator_
print(f'Random forest algorithm\nbest estimator: {optrf}')
preds = optrf.predict(X_test_norm)
print(f'accuracy: {round(accuracy_score(preds, y_test), 3)}\n')
