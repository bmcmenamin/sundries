import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from model_wrangler.model_wrangler import ModelWrangler

MODEL_NAME = 'model_324'
WORK_DIR = '/Users/mcmenamin/GitHub/sundries/inaturalist'

DATA_DIR = os.path.join(WORK_DIR, 'data')
TEST_FILE = os.path.join(WORK_DIR, 'test2018.json')
TRAIN_FILE = os.path.join(WORK_DIR, 'train2018.json')
VALID_FILE = os.path.join(WORK_DIR, 'val2018.json')


def load_coco_json(fname, is_validation=False, num_to_test=None):

    with open(fname, 'rt') as in_file:
        file_dict = json.load(in_file)

    df_images = pd.DataFrame(file_dict['images'], dtype=str)

    good_pics = [os.path.exists(os.path.join(DATA_DIR, f)) for f in df_images.file_name]
    df_images = df_images[good_pics]

    df_images.id = df_images.id.astype(str)

    df_data = df_images.copy()
    if is_validation:
        df_categories = pd.DataFrame(file_dict['categories'], dtype=str)
        df_mapping = pd.DataFrame(file_dict['annotations'], dtype=str)

        df_mapping.image_id = df_mapping.image_id.astype(str)
        df_mapping.category_id = df_mapping.category_id.astype(str)
        df_categories.id = df_categories.id.astype(str)

        df_data = df_images.merge(df_mapping, left_on='id', right_on='image_id')
        df_data = df_data[['file_name', 'category_id', 'image_id']]

    if num_to_test:
        df_data = df_data.sample(num_to_test)

    return df_data


if __name__ == '__main__'

    batch_size = 32
    val_size = 250

    print('Getting data samples')

    df_val_data = load_coco_json(TRAIN_FILE, is_validation=True, num_to_test=val_size)
    df_test_data = load_coco_json(TEST_FILE, is_validation=False, num_to_test=None)

    print("Loading model from disk")
    model_params = os.path.join(WORK_DIR, MODEL_NAME, 'model_params.pickle')
    model = ModelWrangler.load(model_params)

    print("Running validation on {} samples".format(val_size))
    preds_val = []
    file_list = df_val_data.file_name.tolist()
    for idx in tqdm(range(0, len(file_list), batch_size)):
        _file_batch = [
            os.path.join(DATA_DIR, f)
            for f in file_list[idx: (idx + batch_size)]
        ]
        _pred_batch = model.predict([_file_batch])[0]
        preds_val += list(np.argmax(_pred_batch, axis=1))

    df_val_data['pred_id'] = preds_val
    df_val_data['pred_id'] = df_val_data['pred_id'].astype(str)
    accuracy = 100*(df_val_data['pred_id'] == df_val_data['category_id']).mean()
    print('  Validation accuracy is {:.1f}%'.format(accuracy))


    print("Running predictions on test data")
    preds_test = []
    file_list = df_test_data.file_name.tolist()
    for idx in tqdm(range(0, len(file_list), batch_size)):
        _file_batch = [
            os.path.join(DATA_DIR, f)
            for f in file_list[idx: (idx + batch_size)]
        ]
        _pred_batch = model.predict([_file_batch])[0]
        preds_test += [
            ' '.join([str(i) for i in top3])
            for top3 in list(np.argsort(_pred_batch, axis=1)[:, :3])
        ]

    df_test_data['predicted'] = preds_test
    df_test_data.to_csv('predictions_{}.csv'.format(MODEL_NAME))


