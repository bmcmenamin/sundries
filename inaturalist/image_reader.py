import os
import json
import random

import pandas as pd

from model_wrangler.dataset_managers import BaseDatasetManager, LOGGER


def get_category_mask(df_in, family_col):
    in_cat = random.choice(df_in[family_col].unique())
    bool_mask = df_in[family_col] == in_cat
    return bool_mask


def sample_within_family(df_in, batch_size=32, num_in_family=16, family_col='family'):

    cat_mask = get_category_mask(df_in, family_col)

    num_in_family = min([num_in_family, cat_mask.sum()])
    within_rows = df_in.loc[cat_mask, :].sample(num_in_family)

    remaining = min([batch_size - within_rows.shape[0], (~cat_mask).sum()])
    other_rows = df_in.loc[~cat_mask, :].sample(remaining)

    df_batch = pd.concat([within_rows, other_rows])

    return df_batch


class TrainingPhotoManager(BaseDatasetManager):

    def __init__(self, data_dir, fname, embedding_columns, output_column, grouping_column, cache_size=2056):
        """
        Args:
            data_dir: path to data directory
            fname: name of the json holding data definitions
            embedding_columns: list of columns used to guide embedding layer
            output_column: column with the output for softmax
            grouping_column: column used to define the 'in group' for sample
            cache_size: is the number of samples to cache internally
              from the input generators. This list should be larger than the
              batch sizes used in training because it's what gets randomly
              shuffled across epochs
        """

        self.fname = fname
        self.data_dir = data_dir
        self.embedding_columns = embedding_columns
        self.output_column = output_column
        self.grouping_column = grouping_column

        assert self.output_column not in self.embedding_columns

        self.num_inputs = 1
        self.num_outputs = len(self.embedding_columns) + 1
        self.cache_size = cache_size

        self.X = None
        self.Y = None

        LOGGER.info('Dataset has %d inputs', self.num_inputs)
        LOGGER.info('Dataset has %d embedding columns', len(self.embedding_columns))

    def load_coco_json(self):

        with open(self.fname, 'rt') as in_file:
            file_dict = json.load(in_file)

        df_images = pd.DataFrame(file_dict['images'], dtype=str)
        df_categories = pd.DataFrame(file_dict['categories'], dtype=str)
        df_mapping = pd.DataFrame(file_dict['annotations'], dtype=str)


        good_pics = [os.path.exists(os.path.join(self.data_dir, f)) for f in df_images.file_name]
        df_images = df_images[good_pics]

        df_images.id = df_images.id.astype(str)
        df_mapping.image_id = df_mapping.image_id.astype(str)
        df_mapping.category_id = df_mapping.category_id.astype(str)
        df_categories.id = df_categories.id.astype(str)

        return df_images, df_mapping, df_categories

    def _create_epoch_data(self):

        df_images, df_mapping, df_categories = self.load_coco_json()

        df_epoch = (
            df_images.
            merge(df_mapping, how='left', left_on='id', right_on='image_id').
            merge(df_categories, how='left', left_on='category_id', right_on='id')
        ).set_index('image_id')

        map_dict = {}
        for col in self.embedding_columns:
            map_dict = {val: key for key, val in enumerate(df_epoch[col].unique())}
            df_epoch[col] = [map_dict[x] for x in df_epoch[col]]

        df_epoch[self.output_column] = df_epoch[self.output_column].astype(int)
        return df_epoch

    def get_next_batch(self, batch_size=32, eternal=False, **kwargs):
        """
        This generator should yield batches of training data

        Args:
            batch_size: int for number of samples in batch
            eternal: Keep pulling samples forever, or stop after an epoch?
                for some data, it's hard to know when an epoch is over so
                you should use eternal and cap the number of batches
        Yields:
            X, Y: lists of input/output samples
        """

        df_epoch = self._create_epoch_data()
        while not df_epoch.empty:

            # Get a training batch
            df_batch = sample_within_family(
                df_epoch,
                batch_size=batch_size,
                num_in_family=batch_size // 2,
                family_col=self.grouping_column
            )

            # Drop the batch from the current epoch
            if not eternal:
                batch_idx_set = set(df_batch.index)
                df_epoch = df_epoch[~df_epoch.index.isin(batch_idx_set)]

            # Read in images
            X = [[os.path.join(self.data_dir, d) for d in df_batch.file_name]]

            _vectorize = lambda x: pd.np.array(x).reshape(-1, 1)
            Y_embed = [_vectorize(df_batch[c].tolist()) for c in self.embedding_columns]
            Y_output = [_vectorize(df_batch[self.output_column].tolist())]

            Y = Y_output + Y_embed

            yield X, Y

