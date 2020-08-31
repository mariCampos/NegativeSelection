
import random
from negative_selection import NegativeSelection
from utils import create_train_test_database, create_csv_file, calculate_accuracy, calculate_matrix_confusion

def simulate(columns, X_train, X_test):
  negative_selection = NegativeSelection(counter=100, dim=21, r=2.1, train_dataset=X_train, test_dataset=X_test)
  negative_selection.train()
  frauds, not_frauds = negative_selection.test()

  # call csv creator
  create_csv_file(columns, frauds, not_frauds)

  y_test = [0.0 for x in X_test]
  y_pred = [x[1] for x in frauds+not_frauds]

  # calculate accuracy
  accuracy = calculate_accuracy(y_test, y_pred)
  print("Accuracy: {}".format(accuracy))

  matrix = calculate_matrix_confusion(y_test, y_pred)
  print("Confusion matrix: {}".format(matrix))

def main():
    columns, X_train, X_test = create_train_test_database('./database/final_base.csv')
    simulate(columns, X_train, X_test)
    

if __name__ == '__main__':
    main()