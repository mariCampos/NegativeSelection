import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import LeaveOneOut

class NegativeSelection():
  def __init__(self, counter, dim, r, train_dataset, test_dataset):
    self.counter = counter # quantity of lymphocytes to be created 
    self.dim = dim # dimensions of the problem
    self.self_tolerants_ALC = [] # set of articial lymphocytes

    self.train_dataset = train_dataset
    self.test_dataset = test_dataset

    self.r = r # r match continuous or euclidian distance
  
  # generate a randomly ALC
  def create_an_ALC(self):
    return np.random.random_sample(size=self.dim)

  # check affinity between created lymphocytes and patterns
  def check_affinity(self, ALC_created, pattern):
    euclidian_distance = distance.euclidean(ALC_created, pattern)
    if euclidian_distance >= self.r:
      return euclidian_distance,True
    return euclidian_distance, False

  # core 
  def _train(self, train_dataset):
    while len(self.self_tolerants_ALC) < self.counter:
      new_ALC = self.create_an_ALC()
      matched = False
      for pattern in train_dataset:
        euclidian_distance, affinity = self.check_affinity(new_ALC, pattern)
        if affinity:
          matched = True
          break
      if not matched:
        self.self_tolerants_ALC.append(new_ALC)

  def train(self):
    loo = LeaveOneOut()
    count_matches_test = 0
    for train_index, test_index in loo.split(self.train_dataset):
      X_train, X_test = self.train_dataset[train_index], self.train_dataset[test_index]
      self._train(X_train)

  # This method will create a two arrays with the tests and set new target value
  def test(self):
    frauds = []
    not_frauds = []
    distance = 0
    for row in self.test_dataset:
      matched = False
      for alc in self.self_tolerants_ALC:
        distance, affinity = self.check_affinity(row, alc)
        if affinity:
          # if affinity with lymphocytes (lymphocytes are not frauds)
          not_frauds.append(np.concatenate(([distance, affinity], row), axis=None))
          matched = True
          break
      if not matched:
        frauds.append(np.concatenate(([distance, affinity], row), axis=None))

    return frauds, not_frauds
