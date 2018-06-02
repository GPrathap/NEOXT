import numpy as np
import tensorflow as tf


class Neotx():
    def __init__(self):
        self.filters_discriminator = [ 5]
        self.filters_generator = [5]
        self.init_depth_of_discriminator = 16
        self.init_depth_of_generator = 512
        self._BATCH_NORM_DECAY = 0.997
        self._BATCH_NORM_EPSILON = 1e-5

    def batch_normalization_layer(self, layer, is_training):
        gamma_init = tf.random_normal_initializer(1., 0.02)
        layer = tf.layers.batch_normalization(inputs=layer, momentum=self._BATCH_NORM_DECAY,
                                       epsilon=self._BATCH_NORM_EPSILON, center=True, scale=True,
                                               gamma_initializer=gamma_init, training=is_training)
        return tf.nn.leaky_relu(layer, 0.2)

    def get_neoxt_conv2d_transpose_layer(self, current_layers, depth, filters, apply_batch_normalization
                                         , is_train, stride=2):
        next_layers = []
        for filter_size, current_layer in zip(filters, current_layers):
            network = tf.layers.conv2d_transpose(current_layer, depth, [filter_size, filter_size]
                                                 , strides=(stride, stride)
                                                 , padding='SAME', activation=None)
            if apply_batch_normalization:
                network = self.batch_normalization_layer(network, is_train)
            next_layers.append(network)
        return next_layers

    def get_neoxt_generator_reshape_layer(self, inputs, depth, filters, apply_batch_normalization, is_train):
        next_layers = []
        for _ in filters:
            network = tf.layers.dense(inputs, depth, activation=tf.identity)
            network = tf.reshape(network, shape=(-1, 1, 1, depth))
            if apply_batch_normalization:
                network = self.batch_normalization_layer(network, is_train)
            next_layers.append(network)
        return next_layers

    def get_neoxt_conv2d_layer(self, current_layers, depth, filters, apply_batch_normalization, is_train, stride=2):
        next_layers = []
        for filter_size, current_layer in zip(filters, current_layers):
            network = tf.layers.conv2d(current_layer, depth, [filter_size, filter_size], strides=(stride, stride)
                                       , padding='SAME',activation=None)
            if apply_batch_normalization:
                network = self.batch_normalization_layer(network, is_train)
            next_layers.append(network)
        return next_layers

    def get_neoxt_conv2d_first_layer(self, inputs, depth, filters, apply_batch_normalization, is_train, stride=2):
        next_layers = []
        for filter_size in filters:
            network = tf.layers.conv2d(inputs, depth, [filter_size, filter_size], strides=(stride, stride)
                                       , padding='SAME',activation=None)
            if apply_batch_normalization:
                network = self.batch_normalization_layer(network, is_train)
            next_layers.append(network)
        return next_layers

    def get_neoxt_features(self, current_layers):
        features = []
        for current_layer in current_layers:
            network = tf.layers.flatten(current_layer)
            features.append(network)
        return features


    def generator(self, inputs, is_train=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):

            h1_layers = self.get_neoxt_generator_reshape_layer(inputs, self.init_depth_of_generator
                                                               , self.filters_generator, False, is_train)

            depth_of_h1 = int(self.init_depth_of_generator/2)
            h1_layers = self.get_neoxt_conv2d_transpose_layer(h1_layers, depth_of_h1
                                                              , self.filters_generator, True, is_train)

            depth_of_h2 = int(depth_of_h1/2)
            h2_layers = self.get_neoxt_conv2d_transpose_layer(h1_layers, depth_of_h2
                                                              , self.filters_generator, True, is_train)

            depth_of_h3 = int(depth_of_h2/2)
            h3_layers = self.get_neoxt_conv2d_transpose_layer(h2_layers, depth_of_h3
                                                              , self.filters_generator, True, is_train)

            depth_of_h4 = int(depth_of_h3/2)
            h4_layers = self.get_neoxt_conv2d_transpose_layer(h3_layers, depth_of_h4
                                                              , self.filters_generator, True, is_train)

            depth_of_h5 = int(depth_of_h4/2)
            h5_layers = self.get_neoxt_conv2d_transpose_layer(h4_layers, depth_of_h5
                                                              , self.filters_generator, True, is_train)

            depth_of_h6 = int(depth_of_h5/2)
            h6_layers = self.get_neoxt_conv2d_transpose_layer(h5_layers, depth_of_h6
                                                              , self.filters_generator, True, is_train)
            depth_of_h7 = int(depth_of_h6 / 2)
            h7_layers = self.get_neoxt_conv2d_transpose_layer(h6_layers, depth_of_h7
                                                              , self.filters_generator, True, is_train, stride=1)

            net_h8 = tf.layers.conv2d_transpose(h7_layers[0], 3, [1, 1], strides=(1, 1), padding='SAME'
                                                , activation=tf.identity)
            logits = net_h8
            net_h8 = tf.nn.tanh(net_h8)
        return net_h8, logits


    def discriminator(self, inputs, is_train=True, reuse=False):
        feature_set = []
        with tf.variable_scope("discriminator", reuse=reuse):
            net_h0 = self.get_neoxt_conv2d_first_layer(inputs, self.init_depth_of_discriminator
                                                       , self.filters_discriminator, False, is_train)

            depth_of_h1 = self.init_depth_of_discriminator*2
            net_h1 = self.get_neoxt_conv2d_layer(net_h0, depth_of_h1, self.filters_discriminator, True
                                                 , is_train)

            depth_of_h2 = depth_of_h1*2
            net_h2 = self.get_neoxt_conv2d_layer(net_h1, depth_of_h2, self.filters_discriminator, True
                                                 , is_train)

            depth_of_h3 = depth_of_h2*2
            net_h3 = self.get_neoxt_conv2d_layer(net_h2, depth_of_h3, self.filters_discriminator, True
                                                 , is_train)
            feature1 = tf.concat(self.get_neoxt_features(net_h3), axis=1)
            #feature1 = tf.nn.dropout(feature1, keep_prob=0.85)
            feature_set.append(feature1)

            depth_of_h4 = depth_of_h3 * 2
            net_h4 = self.get_neoxt_conv2d_layer(net_h3, depth_of_h4, self.filters_discriminator, True
                                                 , is_train)

            feature2 = tf.concat(self.get_neoxt_features(net_h4), axis=1)
            #feature2 = tf.nn.dropout(feature2, keep_prob=0.85)
            feature_set.append(feature2)

            depth_of_h5 = depth_of_h4 * 2
            net_h5 = self.get_neoxt_conv2d_layer(net_h4, depth_of_h5, self.filters_discriminator, True
                                                 , is_train)
            feature3 = tf.concat(self.get_neoxt_features(net_h5), axis=1)
            #feature3 = tf.nn.dropout(feature3, keep_prob=0.85)
            feature_set.append(feature3)

            #depth_of_h6 = depth_of_h5 * 2
            #net_h6 = self.get_neoxt_conv2d_layer(net_h5, depth_of_h6, self.filters_discriminator, True
            #                                     , is_train, stride=1)
            #feature_set.append(tf.concat(self.get_neoxt_features(net_h6), axis=1))

            feature = tf.concat(feature_set, axis=1)
            feature = tf.nn.dropout(feature, keep_prob=0.85)
            net_h7 = tf.layers.dense(feature, 1, activation=tf.identity)
            logits = net_h7
            net_h7 = tf.nn.sigmoid(net_h7)

        return net_h7, logits, feature
