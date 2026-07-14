import warnings
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")


df = pd.read_csv('zoo.data', header=None)

v = df.iloc[:, -1].values
y, c = pd.factorize(v, sort=True)
X = df.iloc[:, 1:-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0, stratify=y)



scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
n_features = X_train_scaled.shape[1]

print('\n Numbers of training samples =', X_train_scaled.shape[0])
print('\n Numbers of features =', n_features)

models = [LogisticRegression(class_weight='balanced', random_state=0),
          SVC(class_weight='balanced', random_state=0),
          RandomForestClassifier(class_weight='balanced', random_state=0),      
          MLPClassifier(random_state=0)]

models_names = ['Logistic Regression',
                'SVM',
                'Random Forest',
                'MLP']

models_hparametes = [{'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1]},        
                     {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1], 'kernel': ['linear', 'rbf']},
                     {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5]},
                     {'hidden_layer_sizes': [n_features, math.floor(n_features/2), n_features*2], \
                      'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01, 0.1]}                   
                     ]


trained_models = []
validation_performance = []

for model, model_name, hparameters in zip(models, models_names, models_hparametes):
    print('\n ', model_name)
    clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='balanced_accuracy', cv=3)
    clf.fit(X_train_scaled, y_train)
    trained_models.append((model_name, clf.best_estimator_))
    print('Best hiper-parameters:  ', clf.best_params_)
    print('Best balanced accuracy:  ', clf.best_score_)
    validation_performance.append(clf.best_score_)




print('\n..................................................................................')
best_model_index = np.argmax(validation_performance)
final_model = trained_models[best_model_index][1]
print('Best model: ', trained_models[best_model_index][0])
print('\n With hiper-parameters: ', final_model.get_params())
print('\n..................................................................................')


final_model.fit(X_train_scaled, y_train)




X_test_scaled = scaler.transform(X_test)


y_pred = final_model.predict(X_test_scaled)



print('\n/------------------------------------------------------------------------------ /')
print('Final Testing Results')
print('/------------------------------------------------------------------------------ /')

print('\nNumbers of testing samples =', X_test_scaled.shape[0])
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Balanced Accuracy: ', balanced_accuracy_score(y_test, y_pred))


print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
