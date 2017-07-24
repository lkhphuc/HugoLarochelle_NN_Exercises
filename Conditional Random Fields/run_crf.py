import numpy as np
import os
import sys
import fcntl
import copy
from string import Template
import mlpython.datasets.store as dataset_store
import mlpython.mlproblems.generic as mlpb
from crf import LinearChainCRF

sys.argv.pop(0);	# Remove first argument

# Check if every option(s) from parent's script are here.
if 4 != len(sys.argv):
    print "Usage: python run_crf.py lr dc L2 L1 "
    print ""
    print "Ex.: python run_crf.py 0.1 0 0 0"
    sys.exit()

# Set the constructor
str_ParamOption = "lr=" + sys.argv[0] + ", " + "dc=" + sys.argv[1] + ", " + "L2=" + sys.argv[2] + ", " + "L1=" + sys.argv[3] 
str_ParamOptionValue = sys.argv[0] + "\t" + sys.argv[1] + "\t" + sys.argv[2] + "\t" + sys.argv[3] 
try:
    objectString = 'myObject = LinearChainCRF(n_epochs=1,' + str_ParamOption + ')'
    exec objectString
    #code = compile(objectString, '<string>', 'exec')
    #exec code
except Exception as inst:
    print "Error while instantiating LinearChainCRF (required hyper-parameters are probably missing)"
    print inst

print "Loading dataset..."

import ocr_letters_sequential as mldataset
load_to_memory = True
# Try to find dataset in MLPYTHON_DATASET_REPO
name = 'ocr_letters_sequential'
import os
repo = os.environ.get('MLPYTHON_DATASET_REPO')
if repo is None:
    raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/' + name
        
all_data = mldataset.load(dataset_dir,load_to_memory=load_to_memory)

train_data, train_metadata = all_data['train']
valid_data, valid_metadata = all_data['valid']
test_data, test_metadata = all_data['test']

import mlpython.mlproblems.generic as mlpb
trainset = mlpb.MLProblem(train_data,train_metadata)
validset = trainset.apply_on(valid_data,valid_metadata)
testset = trainset.apply_on(test_data,test_metadata)

def compute_error_mean_and_sterror(costs):
    classif_errors = np.hstack([ c[0] for c in costs])
    classif_mean = classif_errors.mean()
    classif_sterror = classif_errors.std(ddof=1)/np.sqrt(classif_errors.shape[0])

    nll_errors = [ c[1] for c in costs]
    nll_mean = np.mean(nll_errors)
    nll_sterror = np.std(nll_errors,ddof=1)/np.sqrt(len(nll_errors))

    return classif_mean, nll_mean, classif_sterror, nll_sterror

print "Training..."
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
    errors = compute_error_mean_and_sterror(costs)
    print 'Epoch',stage,'|',
    print 'Training errors: classif=' + '%.3f'%errors[0]+',', 'NLL='+'%.3f'%errors[1] + ' |',
    outputs, costs = myObject.test(validset)
    errors = compute_error_mean_and_sterror(costs)
    print 'Validation errors: classif=' + '%.3f'%errors[0]+',', 'NLL='+'%.3f'%errors[1]
    error = errors[0]
    if error < best_val_error:
        best_val_error = error
        best_it = stage
        n_incr_error = 0
        best_model = copy.deepcopy(myObject)

outputs_tr,costs_tr = best_model.test(trainset)
outputs_v,costs_v = best_model.test(validset)
outputs_t,costs_t = best_model.test(testset)

# Preparing result line
str_modelinfo = str(best_it) + '\t'
train = ""
valid = ""
test = ""
# Get average of each costs
train_errors = compute_error_mean_and_sterror(costs_tr)
valid_errors = compute_error_mean_and_sterror(costs_v)
test_errors = compute_error_mean_and_sterror(costs_t)
str_header += 'train1\tstderr\tvalid1\tstderr\ttest1\tstderr\ttrain2\tstderr\tvalid2\tstderr\ttest2\tstderr'
str_modelinfo += str(train_errors[0]) + '\t' + str(train_errors[2]) + '\t' + str(valid_errors[0]) + '\t' + str(valid_errors[2]) + '\t' + str(test_errors[0]) + '\t' + str(test_errors[2]) + '\t'
str_modelinfo += str(train_errors[1]) + '\t' + str(train_errors[3]) + '\t' + str(valid_errors[1]) + '\t' + str(valid_errors[3]) + '\t' + str(test_errors[1]) + '\t' + str(test_errors[3]) 
str_header += '\n'
result_file = 'results_crf_ocr_letters_sequential.txt'

# Preparing result file
header_line = ""
header_line += 'lr\tdc\tL2\tL1\t'
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

