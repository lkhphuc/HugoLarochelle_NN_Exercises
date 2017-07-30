import sys
sys.argv.pop(0);	# Remove first argument

# Check if every option(s) from parent's script are here.
if 7 != len(sys.argv):
    print "Usage: python run_stacked_autoencoders_nnet.py lr dc sizes pretrain_lr pretrain_n_epochs pretrain_noise_prob seed"
    print ""
    print "Ex.: python run_stacked_autoencoders_nnet.py 0.01 0 [200,100] 0.01 10 0.1 1234"
    sys.exit()

import os
import itertools
import numpy as np
import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import mlpython.mlproblems.generic as mlpb
from nnet import NeuralNetwork
from autoencoder import Autoencoder

# Set the constructor
lr= float(sys.argv[0])
dc= float(sys.argv[1])
exec("sizes=" + sys.argv[2])
pretrain_lr = float(sys.argv[3])
pretrain_n_epochs = int(sys.argv[4])
pretrain_noise_prob = float(sys.argv[5])
seed = int(sys.argv[6])
str_ParamOptionValue = sys.argv[0] + "\t" + sys.argv[1] + "\t" + sys.argv[2] + "\t" + sys.argv[3] + "\t" + sys.argv[4] + "\t" + sys.argv[5] + "\t" + sys.argv[6]

print "Loading dataset..."
trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')

print 'Pretraining hidden layers...'
pretrained_Ws = []
pretrained_bs = []
for i,hidden_size in enumerate(sizes):

    # Defining function that maps dataset
    # into last trained representation
    def new_representation(example,metadata):
        ret = example[0]
        for W,b in itertools.izip(pretrained_Ws,pretrained_bs):
            ret = 1./(1+np.exp(-(b + np.dot(ret,W))))
        return ret

    # Create greedy module training set using PreprocessedProblem
    if i == 0:
        new_input_size = trainset.metadata['input_size']
    else:
        new_input_size = sizes[i-1]
    pretraining_trainset = mlpb.PreprocessedProblem(trainset,preprocess=new_representation,
                                                    metadata={'input_size':new_input_size})

    # Train greedy module
    print '... hidden layer ' + str(i+1),
    new_layer = Autoencoder(n_epochs = pretrain_n_epochs,
                            hidden_size = hidden_size,
                            lr = pretrain_lr,
                            noise_prob = pretrain_noise_prob,
                            seed=seed
                            )
    new_layer.train(pretraining_trainset)
    print ' DONE'

    pretrained_Ws += [new_layer.W]
    pretrained_bs += [new_layer.b]

pretrained_Ws += [np.zeros((sizes[-1],len(trainset.metadata['targets'])))]
pretrained_bs += [np.zeros((len(trainset.metadata['targets'],)))]
    
# Construct neural network, with pretrained parameters
myObject = NeuralNetwork(n_epochs=1,
                         lr=lr,
                         dc=dc,
                         sizes=sizes,
                         seed=seed,
                         parameter_initialization=(pretrained_bs,pretrained_Ws))

print "Fine-tuning..."
# Early stopping code
best_val_error = np.inf
best_it = 0
str_header = 'best_it\t'
look_ahead = 5
n_incr_error = 0
for stage in range(1,500+1,1):
    if not n_incr_error < look_ahead:
        break
    myObject.n_epochs = stage
    myObject.train(trainset)
    n_incr_error += 1
    outputs, costs = myObject.test(trainset)
    errors = np.mean(costs,axis=0)
    print 'Epoch',stage,'|',
    print 'Training errors: classif=' + '%.3f'%errors[0]+',', 'NLL='+'%.3f'%errors[1] + ' |',
    outputs, costs = myObject.test(validset)
    errors = np.mean(costs,axis=0)
    print 'Validation errors: classif=' + '%.3f'%errors[0]+',', 'NLL='+'%.3f'%errors[1]
    error = errors[0]
    if error < best_val_error:
        best_val_error = error
        best_it = stage
        n_incr_error = 0
        best_model = copy.deepcopy(myObject)

outputs_tr,costs_tr = best_model.test(trainset)
columnCount = len(costs_tr.__iter__().next())
outputs_v,costs_v = best_model.test(validset)
outputs_t,costs_t = best_model.test(testset)

# Preparing result line
str_modelinfo = str(best_it) + '\t'
train = ""
valid = ""
test = ""
# Get average of each costs
for index in range(columnCount):
    train = str(np.mean(costs_tr,axis=0)[index])
    valid = str(np.mean(costs_v,axis=0)[index])
    test = str(np.mean(costs_t,axis=0)[index])
    str_header += 'train' + str(index+1) + '\tvalid' + str(index+1) + '\ttest' + str(index+1)
    str_modelinfo += train + '\t' + valid + '\t' + test
    if ((index+1) < columnCount): # If not the last
        str_header += '\t'
        str_modelinfo += '\t'
str_header += '\n'
result_file = 'results_stacked_autoencoders_nnet_ocr_letters.txt'

# Preparing result file
header_line = ""
header_line += 'lr\tdc\tsizes\tpretrain_lr\tpretrain_n_epochs\tpretrain_noise_prob\tseed\t'
header_line += str_header
if not os.path.exists(result_file):
    f = open(result_file, 'w')
    f.write(header_line)
    f.close()

# Look if there is optional values to display
if str_ParamOptionValue == "":
    model_info = [str_modelinfo]
else:
    model_info = [str_ParamOptionValue,str_modelinfo]

line = '\t'.join(model_info)+'\n'
f = open(result_file, "a")
fcntl.flock(f.fileno(), fcntl.LOCK_EX)
f.write(line)
f.close() # unlocks the file

