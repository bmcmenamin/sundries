import sys
sys.path.append('/home/mcmenamin/model_wrangler')

import os

from model_wrangler.model_wrangler import ModelWrangler
from model_wrangler.model.corral.text_lstm import TextLstmModel
from model_wrangler.dataset_managers import SequentialDatasetManager

TRAIN_FILE = os.path.join(os.path.curdir, 'data', 'train_data.txt')
TEST_FILE = os.path.join(os.path.curdir, 'data', 'test_data.txt')

WINDOW_LENGTH = 128

def lazy_file(fname):
    with open(fname, 'rt') as file:
        for line in file:
            yield line


LSTM_PARAMS = {
    'name': 'moby_model_2',
    'path': './moby_model_2',
    'graph': {
        'in_sizes': [[WINDOW_LENGTH, 1]],
        'recurr_params': [
            {
                'units': 1024,
                'dropout': 0.01,
            },
            {
                'units': 1024,
                'dropout': 0.01,
            },
        ],
        'out_sizes': [1],
    },
    'train': {
        'num_epochs': 3,
        'epoch_length': 1500000,
        'batch_size': 32,
	'learning_rate': 0.0001
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

lstm_model = ModelWrangler(TextLstmModel, LSTM_PARAMS)
lstm_model.add_data(*dm_list)
lstm_model.add_train_params(LSTM_PARAMS['train'])
lstm_model.train()
