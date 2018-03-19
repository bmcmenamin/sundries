import sys
sys.path.append('/home/mcmenamin/model_wrangler')

import os

from model_wrangler.model_wrangler import ModelWrangler

from conv_embed_model import ConvEmbedModel
from image_reader import TrainingPhotoManager

ROOT_DIR = '/mnt/disks/image-data/'
DATA_DIR = ROOT_DIR
TRAIN_FILE = os.path.join(ROOT_DIR, 'train2018.json')
VALID_FILE = os.path.join(ROOT_DIR, 'val2018.json')

EMBED_COLS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
GROUPING_COL = 'family'
OUTPUT_COL = 'category_id'


MODEL_PARAMS = {
    'name': 'model_318',
    'path': './model_318',

    'graph': {
        'in_sizes': [[500, 500, 1]],
        'preinception_params': [
            {
                'num_units': 64,
                'kernel': [7, 7],
                'strides': [2, 2],
                'pool_size': [2, 2],
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
            },
            {
                'num_units': 175,
                'kernel': [3, 3],
                'strides': [1, 1],
                'pool_size': [2, 2],
                'bias': True,
                'activation': 'relu',
                'activity_reg': {'l1': 0.1},
            },
        ],
        'inception_params': [
            {
                'size_1x1_output': 16,
                'downsample': None,
                'total_output_units': 250,
            },
            {
                'size_1x1_output': 16,
                'downsample': 2,
                'total_output_units': 500,
            },
            {
                'size_1x1_output': 16,
                'downsample': None,
                'total_output_units': 500,
            },
            {
                'size_1x1_output': 16,
                'downsample': None,
                'total_output_units': 500,
            },
            {
                'size_1x1_output': 16,
                'downsample': None,
                'total_output_units': 500,
            },
            {
                'size_1x1_output': 16,
                'downsample': 2,
                'total_output_units': 800,
            },
            {
                'size_1x1_output': 16,
                'downsample': None,
                'total_output_units': 800,
            },
            {
                'size_1x1_output': 16,
                'downsample': None,
                'total_output_units': 800,
            },
        ],
        'embed_params': {
            'num_units': 300,
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
