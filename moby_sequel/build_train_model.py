import sys
sys.path.append('/Users/mcmenamin/GitHub/model_wrangler')

import os

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.model.corral.text_lstm import TextLstmModel
from model_wrangler.dataset_managers import SequentialDatasetManager

TRAIN_FILE = os.path.join(os.path.curdir, 'data', 'train_data.txt')
TEST_FILE = os.path.join(os.path.curdir, 'data', 'test_data.txt')

MAX_STRING_SIZE = 4096
WINDOW_LENGTH = 256

def lazy_file(fname):
    with open(fname, 'rt') as file:
        for line in file:
            yield line


LSTM_PARAMS = {
    'name': 'moby_model',
    'path': './moby_model',
    'graph': {
        'max_string_size': MAX_STRING_SIZE,
        'in_sizes': [[WINDOW_LENGTH, 1]],
        'dense_params': [
            {
                'num_units': 256,
                'bias': True,
                'activation': 'relu',
                'dropout': 0.1
            },
        ],
        'recurr_params': [
            {
                'units': 256,
                'activation': 'relu',
            },
            {
                'units': 1024,
                'activation': 'relu',
            }
        ],
        'out_sizes': [1],
    },
    'train': {
        'num_epochs': 30,
        'epoch_length': 10000,
        'batch_size': 32
    }
}

dm_list = [
    SequentialDatasetManager(
        [lazy_file(f)],
        in_win_len=WINDOW_LENGTH,
        out_win_len=1,
        cache_size=128
    )
    for f in [TRAIN_FILE, TEST_FILE]
]

lstm_model = ModelWrangler(TextLstmModel, LSTM_PARAMS)
lstm_model.add_data(*dm_list)
lstm_model.train()
