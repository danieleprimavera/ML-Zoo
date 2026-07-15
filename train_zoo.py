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


#EDA

df = pd.read_csv('zoo.data', header=None)

# Estrazione delle etichette (target) e delle caratteristiche (features):
v = df.iloc[:, -1].values
y, c = pd.factorize(v, sort=True)
X = df.iloc[:, 1:-1].values

# Partizionamento del dataset in training set e test set:
# L'opzione stratify=y garantisce il bilanciamento delle classi in entrambe le porzioni
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0, stratify=y)


# Standardizzazione delle caratteristiche:
# Applichiamo lo scalamento poiché gli algoritmi scelti (es. SVM e MLP) sono sensibili alle differenze di scala:
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
n_features = X_train_scaled.shape[1]  # Numero di attributi, utile per configurare il numero di unità nello strato nascosto dell'MLP

print('\n Numero di training sample =', X_train_scaled.shape[0])
print('\n Numero di feature =', n_features)


#modelli

# Gestione del forte sbilanciamento tramite class_weight='balanced'; random_state=0 assicura la riproducibilità
models = [LogisticRegression(class_weight='balanced', random_state=0),
          SVC(class_weight='balanced', random_state=0),
          RandomForestClassifier(class_weight='balanced', random_state=0),      
          MLPClassifier(random_state=0)] # Nota: MLPClassifier non supporta nativamente il bilanciamento pesato delle classi

models_names = ['Logistic Regression',
                'SVM',
                'Random Forest',
                'MLP']

models_hparametes = [{'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1]},           # Regressione Logistica: testiamo regolarizzazioni L1/L2 e diverse intensità di penalizzazione C
                     {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1], 'kernel': ['linear', 'rbf']}, # SVM: esploriamo confini lineari e non lineari (RBF) con differenti parametri di regolarizzazione C
                     {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5]},  # Random Forest: variamo il numero di alberi e la loro profondità per controllare la complessità del modello
                     {'hidden_layer_sizes': [n_features, math.floor(n_features/2), n_features*2], \
                      'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01, 0.1]}                    # MLP: valutiamo differenti dimensioni dello strato nascosto, penalizzazioni L2 (alpha) e tassi di apprendimento iniziali
                     ]


trained_models = []
validation_performance = []

for model, model_name, hparameters in zip(models, models_names, models_hparametes):
    print('\n ', model_name)
    clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='balanced_accuracy', cv=3)
    clf.fit(X_train_scaled, y_train)
    trained_models.append((model_name, clf.best_estimator_))
    print('I valori migliori degli iper-parametri sono:  ', clf.best_params_)
    print('Accuracy:  ', clf.best_score_)
    validation_performance.append(clf.best_score_)




# Selezione e identificazione del miglior modello basato sui punteggi di validazione

print('\n..................................................................................')
best_model_index = np.argmax(validation_performance)
final_model = trained_models[best_model_index][1]
print('Ho scelto come miglior modello : ', trained_models[best_model_index][0])
print('\n I cui iper-parametri sono: ', final_model.get_params())
print('\n..................................................................................')


# Addestramento finale del modello selezionato sull'intero set di training

final_model.fit(X_train_scaled, y_train)


# Fase di test e valutazione delle performance

# Applicazione della standardizzazione (StandardScaler) ai dati di test

X_test_scaled = scaler.transform(X_test)


# Predizione delle etichette e calcolo delle metriche di valutazione

y_pred = final_model.predict(X_test_scaled)


# Stampa dei risultati finali di test

print('\n/------------------------------------------------------------------------------ /')
print('RISULTATI Finali del Testing')
print('/------------------------------------------------------------------------------ /')

print('\nNumero di testing samples =', X_test_scaled.shape[0])
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Balanced Accuracy: ', balanced_accuracy_score(y_test, y_pred))


print('\nMatrice di Confusione:')
print(confusion_matrix(y_test, y_pred))
print('\nReport di Classificazione:')
print(classification_report(y_test, y_pred))
