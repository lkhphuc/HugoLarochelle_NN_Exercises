import os
import itertools
import numpy as np
import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import mlpython.mlproblems.generic as mlpb
#from rbm import RBM
from autoencoder import Autoencoder

print "Loading dataset..."
trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')
print "Train autoencoder for 10 iterations... (this might take a few minutes)"
aa = Autoencoder(n_epochs = 10,
                 hidden_size = 200,
                 lr = 0.01,
                 noise_prob = 0.25,
                 seed=1234
                 )

aa.train(mlpb.SubsetFieldsProblem(trainset))
aa.show_filters()
