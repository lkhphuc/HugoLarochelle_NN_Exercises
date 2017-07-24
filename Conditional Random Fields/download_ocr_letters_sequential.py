import ocr_letters_sequential as mldataset
import os
repo = os.environ.get('MLPYTHON_DATASET_REPO')
if repo is None:
    raise ValueError('environment variable MLPYTHON_DATASET_REPO is not defined')
dataset_dir = os.environ.get('MLPYTHON_DATASET_REPO') + '/ocr_letters_sequential'

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
mldataset.obtain(dataset_dir)
