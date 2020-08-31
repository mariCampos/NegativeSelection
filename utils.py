
import pandas as pd
import csv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_accuracy(y_test, y_pred):
    return accuracy_score(y_pred,y_test)

def calculate_matrix_confusion(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)

def create_train_test_database(path):
    dataset = pd.read_csv(path)
    columns = list(dataset.columns)

    scaler = MinMaxScaler()
    dataset[columns] = scaler.fit_transform(dataset)

    X_train = dataset.loc[dataset['target'] == 1]
    X_train = X_train.iloc[:,1:]
    X_train = X_train.to_numpy()

    X_test = dataset.loc[dataset['target'] == 0]
    X_test = X_test.iloc[:,1:]
    X_test = X_test.to_numpy()
    return columns, X_train, X_test

def create_csv_file(columns, frauds, not_frauds):
  now = datetime.now()
  with open('./database/final_classification_{}.csv'.format(now), 'w', newline='') as file:
    writer = csv.writer(file)
    new_columns = ['distance'] + columns
    rows = []
    rows.append(new_columns)
    rows = rows + frauds + not_frauds
    writer.writerows(rows)