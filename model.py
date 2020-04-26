#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author: lea.cgh
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import tensorflow as tf
import collections


class Layers(object):
    @staticmethod
    def cross(x0, x, name):
        with tf.variable_scope(name):
            _, hidden_size = x0.get_shape().as_list()
            w = tf.get_variable("weight", [hidden_size, 1], initializer=tf.glorot_normal_initializer())
            b = tf.get_variable("bias", [hidden_size], initializer=tf.zeros_initializer())
            # ? X H * H X 1 ==> ? X 1
            xw = tf.matmul(x, w, name="xw")
            return x0 * xw + (b + x)

    @staticmethod
    def fully_connected(x, units, name, use_bias=True):
        with tf.variable_scope(name):
            _, hidden_size = x.get_shape().as_list()
            w = tf.get_variable("weight", [hidden_size, units], initializer=tf.glorot_normal_initializer())
            if use_bias:
                b = tf.get_variable("bias", [units], initializer=tf.zeros_initializer())
                return tf.nn.xw_plus_b(x, w, b)
            else:
                return tf.matmul(x, w)

    @staticmethod
    def context_fully_connected(x, context, units, context_size, name, use_bias=True):
        with tf.variable_scope(name):
            _, hidden_size = x.get_shape().as_list()
            weight = tf.get_variable("weight", [context_size, hidden_size, units], initializer=tf.glorot_normal_initializer())
            weight = tf.nn.embedding_lookup(weight, context)
            outputs = tf.reshape(tf.matmul(tf.expand_dims(x, axis=1), weight), [-1, units])
            if use_bias:
                bias = tf.get_variable("bias", [context_size, units], initializer=tf.zeros_initializer())
                bias = tf.nn.embedding_lookup(bias, context)
                outputs = outputs + bias
            return outputs

    @staticmethod
    def context_cross(x0, x, context, context_size, name):
        with tf.variable_scope(name):
            _, hidden_size = x0.get_shape().as_list()
            weight = tf.get_variable("weight", [context_size, hidden_size, 1], initializer=tf.glorot_normal_initializer())
            weight = tf.nn.embedding_lookup(weight, context)
            bias = tf.get_variable("bias", [context_size, hidden_size], initializer=tf.zeros_initializer())
            bias = tf.nn.embedding_lookup(bias, context)
            # ? X 1 X H * ? X H X 1 ==> ? X 1
            xw = tf.squeeze(tf.matmul(tf.expand_dims(x, axis=1), weight, name="xw"), axis=-1)
            return x0 * xw + (bias + x)


class Model(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def _prase_inputs(self, inputs, context, context_size):
        def get_embedding_lookup_table(name, shape, **kwargs):
            return tf.get_variable(name, shape, kwargs.get("dtype", tf.float32), kwargs.get("initializer", tf.variance_scaling_initializer(1.0, 'fan_out', 'normal')), partitioner=kwargs.get("paritioner", None), trainable=kwargs.get("trainable", True))
        outputs = {}
        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            for name in sorted(self.embedding_config.keys()):
                bucket_size, hidden_size = self.embedding_config[name]
                common_embedding_lookup_table = get_embedding_lookup_table(name + '_common', (bucket_size, hidden_size))
                # CTR
                if self.composition['context_emb']:
                    context_embedding_lookup_table = get_embedding_lookup_table(name + '_context', (bucket_size * context_size, hidden_size // 2))
                    outputs[name] = tf.concat([
                        tf.nn.embedding_lookup(common_embedding_lookup_table, inputs[name]),
                        tf.nn.embedding_lookup(context_embedding_lookup_table, context * bucket_size + tf.to_int64(inputs[name])),
                    ], axis=-1)
                else:
                    outputs[name] = tf.nn.embedding_lookup(common_embedding_lookup_table, inputs[name])
                tf.add_to_collection('embedding_activations', outputs[name])
        if 'context' in inputs:
            outputs['context'] = inputs['context'] * tf.constant(0.2, tf.float32)
        if 'position' in inputs:
            outputs['position'] = inputs['position'] * tf.constant(0.2, tf.float32)
        return outputs, ge_outputs

    def inference(self, inputs, context, context_size, training=True, return_common=False):
        features, ge_features = self._prase_inputs(inputs, context, context_size)
        net = tf.concat(features.values(), axis=-1)
        with tf.variable_scope("Common"):
            deep_net = cross_net = net
            for idx in range(self.layer_count['common_dnn']):
                deep_net = tf.nn.relu(Layers.fully_connected(deep_net, self.hidden_size_base * (idx + 1), "Deep%d" % idx))
            for idx in range(self.layer_count['common_dcn']):
                cross_net = Layers.cross(net, cross_net, "Cross%d" % idx)
            with tf.variable_scope("Logits"):
                common_net = tf.concat([deep_net, cross_net], axis=-1)
                if return_common:
                    return common_net
                common_logits = Layers.fully_connected(common_net, context_size, "Dense")
                if self.composition['js']:
                    common_logits = tf.nn.softmax(common_logits)
        with tf.variable_scope("Context"):
            context_deep_net = deep_net
            for idx in range(self.layer_count['context_dnn'], 0, -1):
                context_deep_net = tf.nn.relu(Layers.context_fully_connected(context_deep_net, context, self.hidden_size_base * idx, context_size, "ContextDeep%d" % idx))
            context_cross_net = cross_net
            for idx in range(self.layer_count['context_dcn'], 0, -1):
                context_cross_net = Layers.context_cross(net, context_cross_net, context, context_size, "ContextCross%d" % idx)
            with tf.variable_scope("Logits"):
                central_bias = tf.get_variable("central_bias", (), tf.float32, tf.zeros_initializer)
                context_net = tf.concat([context_deep_net, context_cross_net], axis=-1)
                context_logits = Layers.context_fully_connected(context_net, context, 1, context_size, "Dense") + central_bias
        return common_logits, context_logits, ge_features

    def build_graph(self, inputs):
        with tf.device("/gpu:0"):
            context, context_size = tf.reshape(tf.to_int64(inputs['context']), [-1]), self.context_size
            with tf.name_scope("Inference"):
                common_logits, context_logits, ge_features = self.inference(inputs, context, context_size)
            with tf.name_scope("GroupVars"):
                group_vars = {'embedding': [], 'bias': [], 'weight': []}
                for variable in tf.trainable_variables():
                    lst = None
                    for name in group_vars:
                        if name in variable.name:
                            lst = group_vars[name]
                            break
                    assert lst is not None, 'Unknown Variable: %s' % variable
                    lst.append(variable)
                group_vars['embedding_activations'] = tf.get_collection('embedding_activations')
                for name in group_vars:
                    var_name_list = ', '.join([var.name for var in group_vars[name]])
                    print("Group Name: {name}, Var Name List: {var_name_list}".format(**locals()))
            with tf.name_scope("Loss"):
                labels = tf.to_float(inputs['label'])
                with tf.variable_scope("CE"):
                    ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=context_logits), name="ce_loss")
                with tf.variable_scope("L1"):
                    if self.loss_scaler['l1_regularization'] == 0.0:
                        l1_loss = tf.constant(0)
                    else:
                        raise NotImplementedError("l1_regularization")
                with tf.variable_scope("L2"):
                    if self.loss_scaler['l2_regularization'] == 0.0:
                        l2_loss = tf.constant(0)
                    else:
                        var_list = group_vars.get('embedding_activations', []) + group_vars.get('weight', [])
                        l2_loss = tf.multiply(self.loss_scaler['l2_regularization'], tf.divide(1.0, len(var_list)) * tf.add_n([tf.nn.l2_loss(var) for var in var_list]), name="l2_loss")
                with tf.variable_scope("JS"):
                    if self.composition['js']:
                        dist = tf.split(common_logits, context_size, axis=-1)
                        js_loss = []
                        for i in range(context_size):
                            for j in range(i+1, context_size):
                                with tf.variable_scope("js_%d_%d" % (i, j)):
                                    reverse_ij_dist = 2. / (dist[i] + dist[j])
                                    js_loss.append(tf.reduce_sum(dist[i] * tf.log(dist[i] * reverse_ij_dist) + dist[j] * tf.log(dist[j] * reverse_ij_dist)))
                        js_loss = tf.multiply(self.loss_scaler['js'], tf.reduce_mean(js_loss))
                    else:
                        js_loss = tf.constant(0.0)
                with tf.variable_scope("total_loss"):
                    total_loss = tf.add_n([ce_loss, js_loss, l2_loss])
            with tf.name_scope("Train"):
                global_step = tf.train.get_or_create_global_step()
                learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rate_values, name="learning_rate")
                train_ops = (tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []) + [
                    tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step)
                ]
                train_op = tf.group(*train_ops)
            with tf.name_scope("Others"):
                scores = tf.reshape(tf.nn.sigmoid(context_logits), [-1])
                labels = tf.reshape(tf.to_int32(inputs['label']), [-1])
                context = tf.reshape(tf.to_int32(inputs['context']), [-1])
            return scores, labels, context, global_step, {'ce': ce_loss, 'js': js_loss, 'l1': l1_loss, 'l2': l2_loss}, learning_rate, train_op
