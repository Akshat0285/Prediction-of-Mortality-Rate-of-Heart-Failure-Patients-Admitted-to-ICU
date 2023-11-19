import csv, os, re, sys, codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, statistics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier

class Classification:
    def __init__(self, clf_opt='lr', impute_opt='mean'):
        self.clf_opt = clf_opt
        self.impute_opt = impute_opt
        self.data = None
        self.labels = None

    def get_data(self):

        # Read the data
        data = pd.read_csv("training_data.csv")
        labels = pd.read_csv("training_data_targets.csv", header=None, names=['target'])['target']
        data = pd.concat([data, labels], axis=1)

        # Remove the 'age' column
        # data = data.drop('age', axis=1)
        
        imputed_data = self.impute_data(data.iloc[:, :-1])
        self.get_class_statistics(labels)
        return imputed_data, labels

    def select_imputer(self):

        # Select the imputer

        if self.impute_opt == 'mean':
            return SimpleImputer(strategy='mean')
        elif self.impute_opt == 'median':
            return SimpleImputer(strategy='median')
        elif self.impute_opt == 'mode':
            return SimpleImputer(strategy='most_frequent')
        elif self.impute_opt == 'knn':
            return KNNImputer()
        else:
            raise ValueError("Invalid imputation option. Choose from 'mean', 'median', 'mode', 'knn'")
    
    def encode_binary_features(self, data):

        # features to encode
        features_to_encode =['hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD']

        for feature in features_to_encode:
            data[f'{feature}_encoded'] = data[feature]

        # drop the original features
        data = data.drop(features_to_encode, axis=1)

        return data

    def impute_data(self, data):
        
        # Encode binary features
        data = self.encode_binary_features(data)

        # Impute the data
        imputer = self.select_imputer()
        return imputer.fit_transform(data)

    def classification_pipeline(self):

        # Select the classifier

# Adaboost classifier
        if self.clf_opt == 'ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
            be2 = LogisticRegression(solver='liblinear', class_weight='balanced') 
            be3 = RandomForestClassifier(max_depth=50, min_samples_split=3, min_samples_leaf=1, criterion = 'gini')
            clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100)
            clf_parameters = {
                'clf__base_estimator': (be1, be2, be3),
                'clf__random_state': (0, 10),
                'clf__learning_rate': (0.1, 1, 10),
                'clf__n_estimators': (50, 100, 200),
                'clf__algorithm': ('SAMME', 'SAMME.R'),
                'clf__random_state': (0, 10),
            }

# Decision Tree classifier

        elif self.clf_opt == 'dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=40) 
            clf_parameters = {
                'clf__criterion': ('gini', 'entropy'), 
                'clf__max_depth': (10, 5, 20, 15 ),
                'clf__min_samples_split': (2, 4, 6),
                'clf__max_features': ('auto', 'sqrt', 'log2'),
                'clf__ccp_alpha': (0.009, 0.01, 0.05, 0.1),
            }

# Logistic Regression classifier

        elif self.clf_opt == 'lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear', class_weight='balanced') 
            clf_parameters = {
                'clf__random_state': (0, 10),
                'clf__penalty': ('l1', 'l2'),
                'clf__C': (0.1, 1, 100),
                'clf__max_iter': (100, 200, 300),
            }

# Linear SVC classifier

        elif self.clf_opt == 'ls':
            print('\n\t### Training Linear SVC Classifier ### \n')
            clf = svm.SVC(kernel='linear', class_weight='balanced', probability=True)  
            clf_parameters = {
                'clf__C': (0.1, 1, 100),
            }

# KNN classifier

        elif self.clf_opt == 'knn':
            print('\n\t### Training KNN Classifier ### \n')
            clf = KNeighborsClassifier()
            clf_parameters = {
                'clf__n_neighbors': (3, 5, 7),
                'clf__weights': ('uniform', 'distance'),
                'clf__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                'clf__leaf_size': (30, 50, 100),
            }

# Random Forest classifier

        elif self.clf_opt == 'rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None, class_weight='balanced')
            clf_parameters = {
                'clf__criterion': ('entropy', 'gini'),       
                'clf__n_estimators': (30, 50, 100),
                'clf__max_depth': (10, 20, 30, 50, 100, 200),
            }

# SVM classifier

        elif self.clf_opt == 'svm':
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced', probability=True)  
            clf_parameters = {
                'clf__C': (0.1, 1, 100),
                'clf__kernel': ('linear', 'rbf', 'polynomial'),
            }

        else:
            print('Select a valid classifier \n')
            sys.exit(0)

# Pipeline

        pipeline = Pipeline([
            ('imputer', self.select_imputer()),
            ('clf', clf),
        ])

        return pipeline, clf_parameters

    def get_class_statistics(self, labels):

        class_statistics = Counter(labels)
        print('\n Class \t\t Number of Instances \n')
        for item in list(class_statistics.keys()):
            print('\t'+str(item)+'\t\t\t'+str(class_statistics[item]))

    def classification(self):

        data, labels = self.get_data()

        # Impute the data
        imputer = self.select_imputer()
        imputed_data = imputer.fit_transform(data)

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_data)
        
        # feature selection
        # selector = SelectKBest(f_classif, k=20)
        # selector = SelectKBest(f_classif, k=30)
        # selector = SelectKBest(f_classif, k=25)
        # selector = SelectKBest(f_classif, k=40)
        # selector.fit(scaled_data, labels)
        # scaled_data = selector.transform(scaled_data)

        #feature selection using mutual information
        # selector = SelectKBest(mutual_info_classif, k=20)
        # selector = SelectKBest(mutual_info_classif, k=20)
        # selector = SelectKBest(mutual_info_classif, k=20)
        # selector = SelectKBest(mutual_info_classif, k=20)
        # selector.fit(scaled_data, labels)
        # scaled_data = selector.transform(scaled_data)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2, random_state=42, stratify=labels)


        # Oversample the training data

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # ros = RandomOverSampler(random_state=42)
        # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        # Classification
        clf, clf_parameters = self.classification_pipeline()

        # Grid Search

        grid = GridSearchCV(clf, clf_parameters, scoring='f1_macro', cv=5)
        grid.fit(X_resampled, y_resampled)
        clf = grid.best_estimator_
        print('\n\t### Best Parameters ### \n')
        print(grid.best_params_)

        # Make predictions on the test set                                                                                                              
        predicted = clf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, predicted)
        precision = precision_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
        f1 = f1_score(y_test, predicted, average = 'macro')
        cm = confusion_matrix(y_test, predicted)

        print('\n\t### Evaluation Metrics ### \n')
        print('Accuracy : ', accuracy, '\n')
        print('Precision', precision, '\n')
        print('Recall : ', recall, '\n')
        print('F1 Score : ', f1, '\n')
        print('Confusion Matrix : \n')
        print(cm)
        print('Classification Report : \n')
        print(classification_report(y_test, predicted))
        print('\n')
