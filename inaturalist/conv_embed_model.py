"""Module sets up Convolutional Embedded model"""

import numpy as np

from keras.applications import InceptionV3, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture, LOGGER
from model_wrangler.model.layers import (
    append_dropout, append_batchnorm, append_dense
)

class ConvEmbedModel(BaseArchitecture):

    # pylint: disable=too-many-instance-attributes

    def __init__(self, params):
        self.inception = None
        super().__init__(params)

    def build_embedder(self, in_layer, dense_params):

        layer_stack = [in_layer]
        for idx, dense_param in enumerate(dense_params):
            with tf.variable_scope('layer_{}'.format(idx)):

                layer_stack.append(
                    append_dense(self, layer_stack[-1], dense_param, 'dense')
                )

                layer_stack.append(
                    append_batchnorm(self, layer_stack[-1], dense_param, 'batchnorm')
                )

                layer_stack.append(
                    append_dropout(self, layer_stack[-1], dense_param, 'dropout')
                )

        return layer_stack[-1]

    def _preprocess_images(self, filenames):

        image_stack = np.vstack([
            self._preprocess_image(str(fname, 'utf-8'))
            for fname in filenames
        ])

        with self.graph.as_default():
            incept_preds = self.inception.predict(image_stack) # BWAAAAAAAMP

        return incept_preds

    def _preprocess_image(self, filename, target_size=[299, 299]):

        try:
            image = load_img(filename, target_size=target_size)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
        except:
            LOGGER.warn('input %s was, like, corrupted or something?', filename)
            image = np.zeros([1] + target_size + [3])

        return image

    def setup_layers(self, params):

        #
        # Load params
        #

        embed_params = params.get('embed_params', {})
        embed_dim = embed_params[-1]['num_units']

        num_output_categories = params.get('num_output_categories', 1)
        num_targets = params.get('num_targets', 1)

        with tf.device('/cpu:0'):
            with tf.variable_scope('inception'):
                self.inception = InceptionV3(
                    include_top=False,
                    pooling='max',
                    weights='imagenet',
                )

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("string", name="input_filenames", shape=[None])
        ]

        with tf.device('/cpu:0'):
            with tf.variable_scope('preprocess'):
                image_batch = tf.py_func(
                    self._preprocess_images,
                    in_layers,
                    tf.float32,
                    name='image_batch'
                )
                image_batch.set_shape([None, 2048])
                image_batch = tf.stop_gradient(image_batch)

        # Rejigger the embeddding layer from inceptionV3
        with tf.device('/gpu:0'):

            with tf.variable_scope('rejiggerer'):
                embeds = self.build_embedder(image_batch, embed_params)

            with tf.variable_scope('decoder'):

                decode_weights = tf.Variable(
                    tf.random_normal([num_output_categories, embed_dim]),
                    name='weights'
                )

                decode_bias = tf.Variable(
                    tf.random_normal([num_output_categories]),
                    name='bias'
                )

                out_layers = [
                    tf.matmul(embeds, decode_weights, transpose_b=True) + decode_bias        
                ]

                target_layers = [
                    tf.placeholder("float", name="target_{}".format(idx), shape=[None, 1])
                    for idx in range(num_targets)
                ]

        # Sum the losses for all the levels of categoirzation

        with tf.variable_scope('losses'):
            embed_loss = tf.reduce_sum([
                tf.contrib.losses.metric_learning.triplet_semihard_loss(
                    tf.reshape(targ, [-1]), embeds
                )
                for targ in target_layers
            ])


            output_loss = tf.reduce_sum(
                tf.contrib.nn.sampled_sparse_softmax_loss(
                    weights=decode_weights,
                    biases=decode_bias,
                    labels=target_layers[0],
                    inputs=embeds,
                    num_sampled=num_output_categories // 4,
                    num_classes=num_output_categories,
                    remove_accidental_hits=True,
                )
            )

            loss = 1000 * embed_loss + output_loss

        tb_scalars = {
            'embed_loss': embed_loss,
            'output_loss': output_loss,
        }

        return in_layers, out_layers, target_layers, embeds, loss, tb_scalars


    def setup_training_step(self, params):
        """Set up loss and training step"""

        learning_rate = params.get('learning_rate', 0.001)
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('gradient/'):
            with tf.control_dependencies(update_ops):
                capped_grads = [
                    (g, v) if g is None
                    else (tf.clip_by_value(g, -1.0, 1.0), v)
                    for g, v in optimizer.compute_gradients(self.loss)
                ]
                train_step = optimizer.apply_gradients(capped_grads)

        return train_step

