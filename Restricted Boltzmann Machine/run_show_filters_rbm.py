import os
import itertools
import numpy as np
import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import mlpython.mlproblems.generic as mlpb
from rbm import RBM
#from autoencoder import Autoencoder

print "Loading dataset..."
trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')
print "Train RBM for 10 iterations... (this might take a few minutes)"
rbm = RBM(n_epochs = 10,
          hidden_size = 200,
          lr = 0.01,
          CDk = 1,
          seed=1234
          )

rbm.train(mlpb.SubsetFieldsProblem(trainset))
rbm.show_filters()

