import sys
sys.path.append('/home/mcmenamin/model_wrangler')

import os
import random
from itertools import cycle

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.model.corral.text_lstm import TextLstmModel
from model_wrangler.dataset_managers import SequentialDatasetManager

ROOT_DIR = '/home/mcmenamin/sundries/moby_sequel/'
MODEL_DIR  = os.path.join(ROOT_DIR, 'mw')

TRAIN_FILE = os.path.join(ROOT_DIR, 'data', 'train_data.txt')
TEST_FILE = os.path.join(ROOT_DIR, 'data', 'test_data.txt')

WINDOW_LENGTH = 140

def dummy_lazy_file(fname):
    for i in cycle([
        ' abcdefghijklmnopqrstuvwxyz ',
        ' ABCDEFGHIJKLMNOPQRSTUVWXYZ ',
        ' 0123456789 ']):
        yield i

def lazy_file(fname):
    with open(fname, 'rt') as file:
        X = [line for line in file]

    random.shuffle(X)
    for line in X:
    	yield line



LSTM_PARAMS = {
    'name': 'moby_model_stride10',
    'path': './moby_model_stride10',
    'graph': {
        'win_length': WINDOW_LENGTH,
        'embed_size': 100,
        'recurr_params': [
            {
                'units': 512,
                'dropout': 0.5,
            },
            {
                'units': 512,
                'dropout': 0.5,
            },
        ],
    },
    'train': {
        'num_epochs': 500000,
        'epoch_length': 1000,
        'batch_size': 10,
        'learning_rate': 0.01,
        'stride': 10
    }
}



dm_list = [
    SequentialDatasetManager(
        [lazy_file(f)],
        in_win_len=WINDOW_LENGTH,
        out_win_len=1,
        cache_size=64
    )
    for f in [TRAIN_FILE, TEST_FILE]
]



param_file = os.path.join(MODEL_DIR, LSTM_PARAMS['path'], 'model_params.pickle')
if os.path.exists(param_file):
    lstm_model = ModelWrangler.load(param_file)
else:
    lstm_model = ModelWrangler(TextLstmModel, LSTM_PARAMS)

lstm_model.add_data(*dm_list)
lstm_model.add_train_params(LSTM_PARAMS['train'])
lstm_model.train()
