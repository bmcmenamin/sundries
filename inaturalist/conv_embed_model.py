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

        return layer_stack[-1]

    def _inception_layer(self, in_layers, params):

        size_1x1_output = params.get('size_1x1_output', 64)
        downsample = params.get('downsample', 1)
        total_output_units = params.get('total_output_units', 2*size_1x1_output)

        assert total_output_units > 2*size_1x1_output

        conv_1x1_default = {
            'num_units': size_1x1_output,
            'kernel': [1, 1],
            'strides': [1, 1],
            'pool_size': [1, 1],
            'bias': True,
            'activation': 'relu',
            'activity_reg': {'l1': 0.1},
        }

        conv_1x1_output = conv_1x1_default.copy()
        conv_1x1_output['stides'] = downsample

        conv_5x5_default = conv_1x1_output.copy()
        conv_5x5_default['kernel'] = [5, 5]
        conv_5x5_default['num_units'] = (total_output_units - 2*size_1x1_output) // 4

        conv_3x3_default = conv_1x1_output.copy()
        conv_3x3_default['kernel'] = [3, 3]
        conv_3x3_default['num_units'] = (total_output_units - 2*size_1x1_output - conv_5x5_default['num_units'])

        pool_3x3_params = {'pool_size': [3, 3], 'padding': 'same'}

        with tf.variable_scope('1x1'):
            filt_1x1_out = append_conv(self, in_layers, conv_1x1_output, 'output')

        with tf.variable_scope('3x3'):
            filt_3x3_pre = append_conv(self, in_layers, conv_1x1_default, 'pre_conv')
            filt_3x3_out = append_conv(self, filt_3x3_pre, conv_3x3_default, 'output')

        with tf.variable_scope('5x5'):
            filt_5x5_pre = append_conv(self, in_layers, conv_1x1_default, 'pre_conv')
            filt_5x5_out = append_conv(self, filt_5x5_pre, conv_5x5_default, 'output')

        with tf.variable_scope('pool'):
            filt_pool_pre = append_maxpooling(self, in_layers, pool_3x3_params, 'pre_conv')
            filt_pool_out = append_conv(self, filt_pool_pre, conv_1x1_output, 'output')

        with tf.variable_scope('ouput'):
            output_stack = [
                tf.concat([filt_1x1_out, filt_3x3_out, filt_5x5_out, filt_pool_out], axis=-1)
            ]

            output_stack.append(
                append_batchnorm(self, output_stack[-1], {}, 'batchnorm')
            )

        return output_stack[-1]


    def build_embedder(self, in_layer, inception_params):
        """Build a stack of layers for mapping an input to an embedding"""

        layer_stack = [in_layer]
        for idx, inception_param in enumerate(inception_params):
            with tf.variable_scope('inception_{}'.format(idx)):
                layer_stack.append(self._inception_layer(layer_stack[-1], inception_param))

        collapse_space = tf.contrib.layers.flatten(
            tf.reduce_mean(layer_stack[-1], axis=[1, 2], keepdims=False)
        )
        concat_dropout = append_dropout(self, collapse_space, {'dropout': 0.4}, 'dropout')

        # Force unit-norm
        #flat = tf.contrib.layers.flatten(collapse_space)
        #norm = tf.norm(flat, ord='euclidean', axis=1, keepdims=True, name='norm')
        
        return concat_dropout

    def _preprocess_images(self, params, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        image_resized = tf.image.resize_image_with_crop_or_pad(
            image_decoded, params['height'], params['width']
        )
        return image_resized

    def setup_layers(self, params):

        #
        # Load params
        #

        in_sizes = params.get('in_sizes', [])
        preinception_params = params.get('preinception_params', [])
        inception_params = params.get('inception_params', [])
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

        # Downsample input before inceptions
        pre_incept_stack = [image_batch]
        with tf.variable_scope('preinception'):
            for idx, param in enumerate(preinception_params):
                with tf. variable_scope('conv_{}'.format(idx)):
                    pre_incept_stack.append(
                        self._conv_layer(pre_incept_stack[-1], param)
                    )

        # Do the inceptions input before inceptions
        conv_stack = self.build_embedder(pre_incept_stack[-1], inception_params)

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


        loss = embed_loss + output_loss

        tb_scalars = {
            'embed_loss': embed_loss,
            'output_loss': output_loss,
        }

        return in_layers, out_layers, target_layers, embeds, loss, tb_scalars


    def setup_training_step(self, params):
        """Set up loss and training step"""

        # Import params
        learning_rate = params.get('learning_rate', 0.01)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)

        return train_step
