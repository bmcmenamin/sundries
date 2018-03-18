import sys
sys.path.append('/home/mcmenamin/model_wrangler')

import os

from model_wrangler.model_wrangler import ModelWrangler

from conv_embed_model import ConvEmbedModel
from image_reader import TrainingPhotoManager

ROOT_DIR = '/Users/mcmenamin/GitHub/sundries/inaturalist/'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_FILE = os.path.join(ROOT_DIR, 'train2018.json')
VALID_FILE = os.path.join(ROOT_DIR, 'val2018.json')

EMBED_COLS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
GROUPING_COL = 'family'
OUTPUT_COL = 'category_id'


MODEL_PARAMS = {
    'name': 'test_model',
    'path': './test_model',

    'graph': {
        'in_sizes': [[500, 500, 1]],
        'inception_params': {
            'sizes': [64, 64]
        },
        'embed_params': {
            'num_units': 30,
            'bias': True,
            'activation': 'tanh',
            'dropout_rate': 0.0,
        },
        'num_output_categories': 8142,
        'num_targets': 1 + len(EMBED_COLS)
    },
    'train': {
        'num_epochs': 5000,
        'epoch_length': 1000,
        'batch_size': 32,
        'learning_rate': 0.01,
    },
    'tensorboard': {
        'scalars': ['embed_loss', 'output_loss']
    }
}


dm_train = TrainingPhotoManager(DATA_DIR, TRAIN_FILE, EMBED_COLS, OUTPUT_COL, GROUPING_COL)
dm_validate = TrainingPhotoManager(DATA_DIR, VALID_FILE, EMBED_COLS, OUTPUT_COL, GROUPING_COL)

param_file = os.path.join(ROOT_DIR, MODEL_PARAMS['path'], 'model_params.pickle')
if os.path.exists(param_file):
    model = ModelWrangler.load(param_file)
else:
    model = ModelWrangler(ConvEmbedModel, MODEL_PARAMS)

model.add_data(dm_train, dm_validate)
model.add_train_params(MODEL_PARAMS['train'])
model.train()
