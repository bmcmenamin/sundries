"""Module sets up Convolutional Siamese model"""

from functools import partial

import tensorflow as tf

from model_wrangler.architecture import BaseArchitecture
from model_wrangler.model.layers import (
    append_dropout, append_batchnorm, append_conv, append_maxpooling, append_dense
)

class ConvEmbedModel(BaseArchitecture):

    # pylint: disable=too-many-instance-attributes

    def _conv_layer(self, in_layers, layer_param):
        layer_stack = [in_layers]

        layer_stack.append(
            append_conv(self, layer_stack[-1], layer_param, 'conv')
            )

        layer_stack.append(
            append_maxpooling(self, layer_stack[-1], layer_param, 'maxpool')
            )

        layer_stack.append(
            append_batchnorm(self, layer_stack[-1], layer_param, 'batchnorm')
            )

        layer_stack.append(
            append_dropout(self, layer_stack[-1], layer_param, 'dropout')
            )

        return layer_stack[-1]

    def build_embedder(self, in_layer, conv_params):
        """Build a stack of layers for mapping an input to an embedding"""

        layer_stack = [in_layer]
        for idx, layer_param in enumerate(conv_params):
            with tf.variable_scope('conv_layer_{}/'.format(idx)):
                layer_stack.append(self._conv_layer(layer_stack[-1], layer_param))

        # Force unit-norm
        flat = tf.contrib.layers.flatten(layer_stack[-1])
        norm = tf.norm(flat, ord='euclidean', axis=1, keepdims=True, name='norm')
        layer_stack.append(tf.divide(flat, norm, name='embed_norm'))

        return layer_stack[-1]

    def _preprocess_images(self, params, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_grey = tf.image.rgb_to_grayscale(image_decoded)
        image_resized = tf.image.resize_image_with_crop_or_pad(
            image_grey, params['height'], params['width']
        )
        return image_resized

    def setup_layers(self, params):

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        conv_params = params.get('conv_params', [])
        embed_params = params.get('embed_params', {})
        embed_dim = embed_params['num_units']

        num_output_categories = params.get('num_output_categories', 1)
        num_targets = params.get('num_targets', 1)

        prepro_params = {'height': in_sizes[0][0], 'width': in_sizes[0][1]}

        if len(in_sizes) != 1:
            raise AttributeError('Embedding net takes one input only!') 

        #
        # Build model
        #

        in_layers = [
            tf.placeholder("string", name="input_{}".format(idx), shape=[None])
            for idx, in_size in enumerate(in_sizes)
        ]

        image_batch = tf.cast(tf.map_fn(
            partial(self._preprocess_images, prepro_params),
            in_layers[0],
            back_prop=False,
            dtype=tf.uint8,
            name='image_batch'
        ), tf.float32)

        conv_stack = self.build_embedder(image_batch, conv_params)
        embeds = append_dense(self, conv_stack, embed_params, 'embed')

        decode_weights = tf.Variable(
            tf.random_normal([num_output_categories, embed_dim]),
            name='decode_weights'
        )
        decode_bias = tf.Variable(
            tf.random_normal([num_output_categories]),
            name='decode_weights'
        )

        out_layers = [
            tf.matmul(embeds, decode_weights, transpose_b=True) + decode_bias        
        ]

        target_layers = [
            tf.placeholder("float", name="target_{}".format(idx), shape=[None, 1])
            for idx in range(num_targets)
        ]

        # Sum the losses for all the levels of categoirzation

        with tf.variable_scope('embed_loss/'):
            embed_loss = tf.reduce_sum([
                tf.contrib.losses.metric_learning.triplet_semihard_loss(
                    tf.reshape(targ, [-1]), embeds
                )
                for targ in target_layers
            ])

        with tf.variable_scope('output_loss/'):

            output_loss = tf.reduce_sum(
                tf.contrib.nn.sampled_sparse_softmax_loss(
                    weights=decode_weights,
                    biases=decode_bias,
                    labels=target_layers[0],
                    inputs=embeds,
                    num_sampled=num_output_categories // 10,
                    num_classes=num_output_categories,
                    remove_accidental_hits=True,
                )
            )


        loss = embed_loss + 10*output_loss

        return in_layers, out_layers, target_layers, embeds, loss


    def setup_training_step(self, params):
        """Set up loss and training step"""

        # Import params
        learning_rate = params.get('learning_rate', 0.01)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)

        return train_step
